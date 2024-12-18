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


def system_template(model_name):
    if model_name == "Qwen/Qwen2.5-7B-Instruct":
        prompt = f"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    else:
        prompt = f"You are a helpful assistant."
    return prompt

def emotion_template(emotion):
    emotion = str(emotion).replace("'", "")
    prompt = "I will classify it into " + emotion + "."
    return prompt

def llamafactory_template(ds):
    text = ds["text"]
    user_prompt1 = f"Please read the following sentence and classify it into one of the three categories based on the emotion expressed: negative, ambiguous, or positive.\n\nYour answer format should be:\nAfter analyzing the whole sentence, I will classify it into [negative/ambiguous/positive].\nExplanation: [Your short explanation]\n\nSentence:\n{text}"
    user_prompt2 = f"Now, please classify it into more detailed emotions. Below is all the emotion categories for your choice.\n[\"amusement\", \"excitement\", \"joy\", \"love\", \"desire\", \"optimism\", \"caring\", \"pride\", \"admiration\", \"gratitude\", \"relief\", \"approval\", \"fear\", \"nervousness\", \"remorse\", \"embarrassment\", \"disappointment\", \"sadness\", \"grief\", \"disgust\", \"anger\", \"annoyance\", \"disapproval\", \"realization\", \"surprise\", \"curiosity\", \"confusion\", \"neutral\"]\nYou can choose at most two emotions and answer it in the list form.\n\nYour answer format should be:\nI will classify it into [emotion list]."
    conversation_dict = {"conversations": [
                            {"from": "human", "value": user_prompt1},
                            {"from": "gpt", "value": ds["total_classify_response"][0]},
                            {"from": "human", "value": user_prompt2},
                            {"from": "gpt", "value": emotion_template(ds["total_emotion"])},
                        ], 
                         "system": system_template("Qwen/Qwen2.5-7B-Instruct")}
    return conversation_dict

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
                # 原始数据无标签
                total_emotion = [item[0] for item in Counter(total_emotion).most_common(2)] # majority voting
                total_classify = [total_classify[0]]
                total_classify_response = [total_classify_response[0]]
                flag = True
            elif emotion[0] in total_emotion and classify[0] in total_classify:
                # 有一个标签被预测到
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
        
    result_llama = [llamafactory_template(ds) for ds in result]
    random.shuffle(result_llama)
    
    split_index = int(len(result_llama) * 0.1)  
    test_data = result_llama[:split_index]    
    train_data = result_llama[split_index:]  
    
    
    with open("llm_data/data_train.json", "w") as train_file:
        json.dump(train_data, train_file, indent=2, ensure_ascii=False)

    with open("llm_data/data_test.json", "w") as test_file:
        json.dump(test_data, test_file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    args = arg_parse()
    actor_name = args.actor_name
    dataset_name = args.dataset_name
    data_enhancement(actor_name, dataset_name)