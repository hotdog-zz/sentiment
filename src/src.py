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

with open("./llm_data/Qwen2.5-7b-instruct-full-sft-testset.json", "r") as f:
    data = json.load(f)

total = 0
count0 = 0
count1 = 0
count2 = 0
for ds in data:
    list1 = ds["emotion"]
    list2 = ds["origin_emotion"]
    if ds["classify"] == ds["origin_classify"][0]:
        total += 1
    if list1 == list2:
        count2 += 1
    elif bool(set(list1) & set(list2)):
        count1 += 1
    else:
        count0 += 1
print(total/len(data))
print(count0/len(data))
print(count1/len(data))
print(count2/len(data))