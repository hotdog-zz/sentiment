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
    response = response_origin.lower()
    temp = response.find("explanation")
    response = response[0:temp]
    response_origin = "\n" + response_origin[temp:]
    if "negative" in response:
        classify.append("negative.")
    if "positive" in response:
        classify.append("positive.")
    if "ambiguous" in response:
        classify.append("ambiguous.")
    if len(classify) == 0:
        classify.append("")
    return classify, response_origin

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
    user_prompt = f"Please read the following sentence and classify it into one of the three categories based on the emotion expressed: negative, ambiguous, or positive.\n\nYour answer format should be:\nAfter analyzing the whole sentence, I will classify it into [negative/ambiguous/positive].\nExplanation: [Your short explanation]\n\nSentence:\n{text}"
    if model_name == "Qwen/Qwen2.5-7B-Instruct":
        prompt = f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\nAfter analyzing the whole sentence, I will classify it into"
    elif model_name == "meta-llama/Meta-Llama-3-8B-Instruct" or model_name == "meta-llama/Llama-3.1-8B-Instruct":
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nAfter analyzing the whole sentence, I will classify it into"
    elif model_name == "THUDM/glm-4-9b-chat":
        prompt = f"[gMASK]<sop><|system|>\nYou are a helpful assistant.<|user|>\n{user_prompt}<|assistant|>\nAfter analyzing the whole sentence, I will classify it into"
    return prompt

def generate_emotion_template(text, model_name, classify):
    user_prompt1 = f"Please read the following sentence and classify it into one of the three categories based on the emotion expressed: negative, ambiguous, or positive.\n\nYour answer format should be:\nAfter analyzing the whole sentence, I will classify it into [negative/ambiguous/positive].\nExplanation: [Your short explanation]\n\nSentence:\n{text}"
    user_prompt2 = f"Now, please classify it into more detailed emotions. Below is all the emotion categories for your choice.\n[\"amusement\", \"excitement\", \"joy\", \"love\", \"desire\", \"optimism\", \"caring\", \"pride\", \"admiration\", \"gratitude\", \"relief\", \"approval\", \"fear\", \"nervousness\", \"remorse\", \"embarrassment\", \"disappointment\", \"sadness\", \"grief\", \"disgust\", \"anger\", \"annoyance\", \"disapproval\", \"realization\", \"surprise\", \"curiosity\", \"confusion\", \"neutral\"]\nYou can choose at most three emotions and answer it in the list form.\n\nYour answer format should be:\nI will classify it into [emotion list]."
    if model_name == "Qwen/Qwen2.5-7B-Instruct":
        prompt = f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{user_prompt1}<|im_end|>\n<|im_start|>assistant\nAfter analyzing the whole sentence, I will classify it into {classify}<|im_end|>\n<|im_start|>user\n{user_prompt2}<|im_end|>\n<|im_start|>assistant\nI will classify it into"
    elif model_name == "meta-llama/Meta-Llama-3-8B-Instruct" or model_name == "meta-llama/Llama-3.1-8B-Instruct":
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt1}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nAfter analyzing the whole sentence, I will classify it into {classify}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt2}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nI will classify it into"
    elif model_name == "THUDM/glm-4-9b-chat":
        prompt = f"[gMASK]<sop><|system|>\nYou are a helpful assistant.<|user|>\n{user_prompt1}<|assistant|>\nAfter analyzing the whole sentence, I will classify it into {classify}<|user|>\n{user_prompt2}<|assistant|>\nI will classify it into"
    print(prompt)
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
            generate_emotion_template(item['text'], model_name, str(clean_response(classify_response)[0][0]) + clean_response(classify_response)[1])
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
            new_data["classify"] = data["classify"]
            new_data["emotion"] = data["emotion"]
            new_data["actor_model"] = actor_name
            new_data["classify_response"] = "After analyzing the whole sentence, I will classify it into " + response
            new_data["origin_classify"] = clean_response(response)[0]
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
    gpu_id = [5, 6, 7]
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
    with open(results_path, "w") as g:
        json.dump(true_data, g, indent=2)


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