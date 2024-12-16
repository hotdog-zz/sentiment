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
from collections import Counter

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--actor_name",
        nargs='+',
        default=["Qwen/Qwen2.5-7B-Instruct"],
    )
    parser.add_argument(
        "--dataset_name",
        nargs='+',
        default=["go_emotion"],
    )
    parser.add_argument("--results_file", type=str, default="llm_data/temp.json")
    args = parser.parse_args()
    return args

def clean_response(response):
    classify = []
    response = response.lower()
    temp = response.find("explanation")
    response = response[0:temp]
    if "negative" in response:
        classify.append("negative")
    if "positive" in response:
        classify.append("positive")
    if "ambiguous" in response:
        classify.append("ambiguous")
    if len(classify) == 0:
        classify.append("")
    return classify


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
        prompt = f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{user_prompt1}<|im_end|>\n<|im_start|>assistant\nAfter analyzing the whole sentence, I will classify it into {classify}.<|im_end|>\n<|im_start|>user\n{user_prompt2}<|im_end|>\n<|im_start|>assistant\nI will classify it into"
    elif model_name == "meta-llama/Meta-Llama-3-8B-Instruct" or model_name == "meta-llama/Llama-3.1-8B-Instruct":
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt1}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nAfter analyzing the whole sentence, I will classify it into {classify}.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt2}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nI will classify it into"
    elif model_name == "THUDM/glm-4-9b-chat":
        prompt = f"[gMASK]<sop><|system|>\nYou are a helpful assistant.<|user|>\n{user_prompt1}<|assistant|>\nAfter analyzing the whole sentence, I will classify it into {classify}.<|user|>\n{user_prompt2}<|assistant|>\nI will classify it into"
    return prompt

def data_enhancement(actor_name, dataset_name):
    result = []
    for ds_name in dataset_name:
        data = []
        for act_name in actor_name:
            temp_name = act_name[act_name.rfind("/")+1:]
            results_path = f"llm_data/{temp_name}-{ds_name}.json"
            with open(f"{results_path}", "r") as f:
                data.append(json.load(f))
        for ds in list(zip(*data)):
            total_classify = []
            total_emotion = []
            total_classify_response = []
            for item in ds:
                classify = item["classify"]
                emotion = item["emotion"]
                total_classify += item["origin_classify"]
                total_emotion += item["origin_emotion"]
                total_classify_response.append(item["classify_response"])
            
            flag = False
            if classify[0] == "ambiguous" and emotion[0] == "neutral":
                total_emotion = [item[0] for item in Counter(total_emotion).most_common(2)]
                total_classify = [total_classify[0]]
                total_classify_response = [total_classify_response[0]]
                flag = True
            elif emotion[0] in total_emotion and classify[0] in total_classify:
                total_emotion = [item[0] for item in Counter(total_emotion).most_common(2)]
                for class_item, response_item in zip(total_classify, total_classify_response):
                    if class_item == classify[0]:
                        total_classify = [class_item]
                        total_classify_response = [response_item]
                flag = True

            if flag:
                result.append({"idx": item["idx"],
                               "dataset": item["dataset"],
                               "text": item["text"],
                               "total_classify": total_classify,
                               "total_classify_response": total_classify_response,
                               "total_emotion": total_emotion})

    with open("llm_data/data_all.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    args = arg_parse()
    actor_name = args.actor_name
    dataset_name = args.dataset_name
    data_enhancement(actor_name, dataset_name)