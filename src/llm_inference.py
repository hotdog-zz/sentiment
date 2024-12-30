import argparse
import json
import os
import re
import pdb
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
import multiprocessing
import time
import random

positive_emotions = ["amusement", "excitement", "joy", "love", "desire", "optimism", "caring", "pride", "admiration", "gratitude", "relief", "approval"]
negative_emotions = ["fear", "nervousness", "remorse", "embarrassment", "disappointment", "sadness", "grief", "disgust", "anger", "annoyance", "disapproval"]
neutral_emotions = ["realization", "surprise", "curiosity", "confusion", "neutral"]

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--actor_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--dataset_name",
        nargs='+',
        default=["go_emotion"],
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--sample_num", type=int, default=1)
    parser.add_argument("--results_file", type=str, default="llm_data/temp.json")
    parser.add_argument("--mode", type=str, default="inference")
    args = parser.parse_args()
    return args

def clean_response(response_origin):
    classify = []
    opinion = []

    response = response_origin.lower()
    temp = response.find("explanation")
    response = response[0:temp]
    response_origin = "\n" + response_origin[temp:]

    if "negative" in response:
        classify.append("negative")
    if "positive" in response:
        classify.append("positive")
    if "ambiguous" in response:
        classify.append("ambiguous")
    if len(classify) == 0:
        classify.append("")
    
    if "support" in response:
        opinion.append("support")
    if "opposition" in response:
        opinion.append("opposition")
    if "neutral" in response:
        opinion.append("neutral")
    if len(opinion) == 0:
        opinion.append("")
    return classify, response_origin, opinion

def clean_emotion_response(response):
    response = response.lower()
    temp = response.find("explanation")
    response = response[0:temp]
    emotion = []
    for word in positive_emotions + negative_emotions + neutral_emotions:
        if word in response:
            if word == "approval":
                temp = word.find("approval")
                if word[temp-1] == "s":
                    continue
            else:
                emotion.append(word)
    return emotion

def generate_classify_template(text, model_name):
    user_prompt = f"Here is an overview of a public opinion event: 姜萍是一名来自涟水中专的女生，她在阿里巴巴达摩举办的高等数学竞赛的预赛中取得了12名的成绩。然而，她在中专里的数学成绩仅为83分的低分，预赛答卷甚至出现了“主=6”,”∑=1/2”等数学名词错误，引起广泛讨论。有些网友支持姜萍，认为中专里也可能有天才，并攻击怀疑的人；有些网友怀疑姜萍，认为是她老师王闰秋帮助作弊，并嘲讽和质疑。\nIn this public opinion event, common internet buzzwords and memes include: jp，姜是姜萍名字的缩写，是中性词；jumping，姜圣则是对她的嘲讽，多为贬义；主=6，∑=1/2，姜萍不等式等内容是对她的质疑和嘲讽，多为贬义。 Please read the following sentence related to this event based on the above context and classify it into one of the three categories based on the emotion expressed: negative, ambiguous, or positive. Please also analyze the sentence's view on 姜萍 and classify it into one of the three catrgories: support, opposition and neutral.\n\nYour answer format should be:\nAfter analyzing the whole sentence, I will classify it into [negative/ambiguous/positive] and [support/opposition/neutral].\nExplanation: [Your short explanation]\n\nSentence:\n{text}"
    if model_name == "Qwen/Qwen2.5-7B-Instruct":
        prompt = f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\nAfter analyzing the whole sentence, I will classify it into"
    elif model_name == "meta-llama/Meta-Llama-3-8B-Instruct" or model_name == "meta-llama/Llama-3.1-8B-Instruct":
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nAfter analyzing the whole sentence, I will classify it into"
    elif model_name == "THUDM/glm-4-9b-chat":
        prompt = f"[gMASK]<sop><|system|>\nYou are a helpful assistant.<|user|>\n{user_prompt}<|assistant|>\nAfter analyzing the whole sentence, I will classify it into"
    return prompt

def generate_emotion_template(text, model_name, classify, opinion, explanation):
    user_prompt1 =f"Here is an overview of a public opinion event: 姜萍是一名来自涟水中专的女生，她在阿里巴巴达摩举办的高等数学竞赛的预赛中取得了12名的成绩。然而，她在中专里的数学成绩仅为83分的低分，预赛答卷甚至出现了“主=6”,”∑=1/2”等数学名词错误，引起广泛讨论。有些网友支持姜萍，认为中专里也可能有天才，并攻击怀疑的人；有些网友怀疑姜萍，认为是她老师王闰秋帮助作弊，并嘲讽和质疑。\nIn this public opinion event, common internet buzzwords and memes include: jp，姜是姜萍名字的缩写，是中性词；jumping，姜圣则是对她的嘲讽，多为贬义；主=6，∑=1/2，姜萍不等式等内容是对她的质疑和嘲讽，多为贬义。 Please read the following sentence related to this event based on the above context and classify it into one of the three categories based on the emotion expressed: negative, ambiguous, or positive. Please also analyze the sentence's view on 姜萍 and classify it into one of the three catrgories: support, opposition and neutral.\n\nYour answer format should be:\nAfter analyzing the whole sentence, I will classify it into [negative/ambiguous/positive] and [support/opposition/neutral].\nExplanation: [Your short explanation]\n\nSentence:\n{text}"
    user_prompt2 = f"Now, please classify it into more detailed emotions. Below is all the emotion categories for your choice.\n[\"amusement\", \"excitement\", \"joy\", \"love\", \"desire\", \"optimism\", \"caring\", \"pride\", \"admiration\", \"gratitude\", \"relief\", \"approval\", \"fear\", \"nervousness\", \"remorse\", \"embarrassment\", \"disappointment\", \"sadness\", \"grief\", \"disgust\", \"anger\", \"annoyance\", \"disapproval\", \"realization\", \"surprise\", \"curiosity\", \"confusion\", \"neutral\"]\nYou can choose at most two emotions and answer it in the list form.\n\nYour answer format should be:\nI will classify it into [emotion list]."
    if model_name == "Qwen/Qwen2.5-7B-Instruct":
        prompt = f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{user_prompt1}<|im_end|>\n<|im_start|>assistant\nAfter analyzing the whole sentence, I will classify it into {classify} and {opinion}.{explanation}<|im_end|>\n<|im_start|>user\n{user_prompt2}<|im_end|>\n<|im_start|>assistant\nI will classify it into"
    elif model_name == "meta-llama/Meta-Llama-3-8B-Instruct" or model_name == "meta-llama/Llama-3.1-8B-Instruct":
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt1}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nAfter analyzing the whole sentence, I will classify it into {classify} and {opinion}.{explanation}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt2}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nI will classify it into"
    elif model_name == "THUDM/glm-4-9b-chat":
        prompt = f"[gMASK]<sop><|system|>\nYou are a helpful assistant.<|user|>\n{user_prompt1}<|assistant|>\nAfter analyzing the whole sentence, I will classify it into {classify} and {opinion}.{explanation}<|user|>\n{user_prompt2}<|assistant|>\nI will classify it into"
    return prompt

def generate_responses(model, sampling_params, prompt):
    outputs = model.generate(prompt, sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    return responses

def batch_inference(
    inference_dataset,
    device_id,
    sampling_params,
    batch_size,
    actor_name,
    model_name,
):
    # device_id = device_id % 8
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    model = LLM(model=actor_name, tensor_parallel_size=1, gpu_memory_utilization=0.87, max_model_len=2048, trust_remote_code=True)
    total_batches = (len(inference_dataset) + batch_size - 1) // batch_size
    results_true = []
    
    # Batch inference
    for batch_idx in tqdm(range(total_batches)):
        start_index = batch_idx * batch_size
        end_index = min((batch_idx + 1) * batch_size, len(inference_dataset))
        batch_items = inference_dataset[start_index: end_index]

        prompts = [
            generate_classify_template(item['text'], model_name)
            for item in batch_items
        ]
        # Generate responses using llama
        batch_classify_responses = generate_responses(
            model=model,
            sampling_params=sampling_params,
            prompt=prompts,
        )
        prompts = [
            generate_emotion_template(item['text'], model_name, str(clean_response(classify_response)[0][0]), str(clean_response(classify_response)[2][0]), clean_response(classify_response)[1])
            for classify_response, item in zip(batch_classify_responses, batch_items)
        ]
        batch_emotion_responses = generate_responses(
            model=model,
            sampling_params=sampling_params,
            prompt=prompts,
        )

        for idx, data in enumerate(batch_items):
            response = batch_classify_responses[idx]
            if response[0] == " ":
                response = response[1:]
            new_data = {}
            new_data["idx"] = data["idx"]
            new_data["dataset"] = data["dataset"]
            new_data["text"] = data["text"]
            new_data["comment_id"] = data["comment_id"]
            new_data["time"] = data["time"]
            # new_data["classify"] = data["classify"]
            # new_data["emotion"] = data["emotion"]
            new_data["actor_model"] = actor_name
            new_data["classify_response"] = "After analyzing the whole sentence, I will classify it into " + response
            new_data["origin_classify"] = clean_response(response)[0]
            new_data["origin_opinion"] = clean_response(response)[2]
            response = batch_emotion_responses[idx]
            if response[0] == " ":
                response = response[1:]
            new_data["emotion_response"] = "I will classify it into " + response
            new_data["origin_emotion"] = clean_emotion_response(response)
            results_true.append(new_data)
    return results_true


def inference_pipeline(
    actor_name,
    model_name,
    ds, 
    temperature, 
    sample_num, 
    results_path,
):
    gpu_id = [6]
    total_gpu = len(gpu_id)
    sampling_params = SamplingParams(max_tokens=512, temperature=temperature)
    # random.shuffle(ds)
    ds = [value for value in ds for i in range(sample_num)]
    inference_dataset = []
    batch_size = (len(ds) - 1) // total_gpu + 1
    for batch_idx in range(total_gpu):
        start_index = batch_idx * batch_size
        end_index = min((batch_idx + 1) * batch_size, len(ds))
        inference_dataset.append(ds[start_index: end_index])
    true_data = []
    false_data = []

    with multiprocessing.Pool(processes=total_gpu) as pool:
        results = [
            pool.apply_async(batch_inference, args=(inference_dataset[device_id], gpu_id[device_id], sampling_params, 128, actor_name, model_name))
            for device_id in range(total_gpu)
        ]
        for r in results:
            result_true = r.get()
            true_data += result_true
    with open(results_path, "w", encoding="utf-8") as g:
        json.dump(true_data, g, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    args = arg_parse()
    actor_name = args.actor_name
    model_name = args.model_name
    dataset_name = args.dataset_name
    temperature = args.temperature
    sample_num = args.sample_num
    mode = args.mode
    for ds_name in dataset_name:
        with open(f"data/{ds_name}/clean_data.json", "r") as f:
            ds = json.load(f)
        temp_name = actor_name[actor_name.rfind("/")+1:]
        results_path = f"llm_data/{temp_name}-{ds_name}.json"
        inference_pipeline(actor_name, model_name, ds, temperature, sample_num, results_path)