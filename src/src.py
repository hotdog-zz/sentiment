import argparse
import json
import os
import re
import pdb
import pandas as pd
import json
from collections import Counter
from collections import defaultdict
import jieba  # 中文分词库，假设文本是中文
import pdb
import math
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import pdb
from statsmodels.tsa.api import VAR
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm
import time
import random

import pandas as pd
from collections import Counter


def confusion_matrix():
    # 模型测试混淆矩阵
    with open("llm_data/Qwen2.5-7b-instruct-full-sft-testset.json", "r") as f:
        data = json.load(f)

    # 初始化计数器
    count_matrix = {
        "positive": {"positive": 0, "ambiguous": 0, "negative": 0},
        "ambiguous": {"positive": 0, "ambiguous": 0, "negative": 0},
        "negative": {"positive": 0, "ambiguous": 0, "negative": 0},
    }

    # 遍历数据并更新计数矩阵
    for ds in data:
        classify = ds["classify"] if ds["classify"] != '' else "ambiguous"
        origin_classify = ds["origin_classify"][0] if ds["origin_classify"][0] != '' else "ambiguous"
        
        # 更新计数
        count_matrix[classify][origin_classify] += 1

    # 将结果转换为DataFrame，便于展示
    df = pd.DataFrame(count_matrix)

    # 打印出表格
    print(df)

    total = 0
    count0 = 0
    count1 = 0
    count2 = 0
    for ds in data:
        list1 = ds["emotion"]
        list2 = ds["origin_emotion"]
        if ds["classify"] == ds["origin_classify"][0]:
            total += 1
        if len(set(list1) & set(list2)) == 2:
            count2 += 1
        elif len(set(list1) & set(list2)) == 1:
            count1 += 1
        else:
            count0 += 1
    print(total/len(data))
    print(count0/len(data))
    print(count1/len(data))
    print(count2/len(data))


def key_word():
    # 情感特征词，使用了改进的tf-idf
    # 加载 JSON 数据
    with open('Qwen2.5-7B-Instruct-douban_reply.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open('Qwen2.5-7B-Instruct-tieba_reply.json', 'r', encoding='utf-8') as f:
        data += json.load(f)
    with open('Qwen2.5-7B-Instruct-weibo_reply.json', 'r', encoding='utf-8') as f:
        data += json.load(f)
    # 转换为 DataFrame
    df = pd.DataFrame(data)

    chinese_stopwords = [
        "的", "了", "在", "是", "我", "有", "和", "你", "他", "她", "它", 
        "我们", "他们", "这", "那", "对", "于", "向", "为", "上", "下", 
        "很", "就", "都", "吧", "说", "也", "还", "但", "如果", "或者", 
        "因为", "所以", "、", "。", "，", "？", "！", "；", "：", "（", "）", 
        "【", "】", "《", "》", "“", "”", "‘", "’", "、", "．", "—", "·", "～"
    ]
    english_stopwords = [
        "i", "me", "you", "he", "she", "it", "we", "they", "a", "an", "the", 
        "and", "or", "but", "if", "in", "on", "at", "by", "for", "to", "with", 
        "of", "that", "this", "who", "which", "where", "how", "when", "as", 
        "be", "been", "being", "am", "is", "are", "was", "were", "been", "being", 
        "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", 
        "the", "for", "nor", "but", "so", "as", "by", "on", "at", "to", "from", 
        "up", "down", "with", "about", "against", "between", "into", "through", 
        "during", "before", "after", "above", "below", "since", "until", "while", 
        "of", "for", "it", "you", "he", "she", "we", "they", "them", "i", "me", 
        "us", "our", "yours", "its", "their", "theirs", "the", "and", "or", "not", 
        "with", "as", "this", "that", "these", "those", "very", "just", "don’t", 
        "should", "now", "too", "only", "own", "same", "self", "all", "can", "will", 
        "up", "down", "out", "into", "or", "but", "if", "then", "more", "less", "so", 
        "what", "when", "where", "why", "how", "each", "every", "some", "any", "no",
        ".", ",", "?", "!", ";", ":", "(", ")", "[", "]", "{", "}", "-", "_", "\"", 
        "'", "`", "‘", "’", "“", "”", "…"
    ]

    # 这里我们将对每一条记录的文本进行分词处理
    df['text_cut'] = df['text'].apply(lambda x: ' '.join(jieba.cut(x)))

    # 用于存储每个情感类别对应的所有文本
    emotion_texts = defaultdict(list)
    emotion_document_count = defaultdict(int)

    # 收集每个情感类别的文本，并统计每个情感类别的文档数量
    for idx, row in df.iterrows():
        emotions = row['origin_emotion']  # 假设每条记录有多个情感标签
        text = row['text_cut']
        for emotion in emotions:
            if emotion not in emotion_texts:
                emotion_document_count[emotion] = 0  # 初始化文档数量
            emotion_texts[emotion].append(text)
            emotion_document_count[emotion] += 1

    # 构建文档集，使用所有情感类别的文本
    all_texts = []
    for texts in emotion_texts.values():
        all_texts += texts

    idf_dict = {}
    for text in all_texts:
        text_dict = set(text.split())
        for word in text_dict:
            if word not in idf_dict.keys():
                idf_dict[word] = 0
            idf_dict[word] += 1

    for emotion, texts in emotion_texts.items():
        # 用情感类别的文本计算 TF，注意此时 IDF 来自于 idf_dict
        split_text = " ".join(texts).split()
        tf = Counter(split_text)
        for word in tf:
            tf[word] *= math.log(len(idf_dict)/idf_dict[word])/len(split_text)
        
        # 按照值排序，获取前 10 个值最大的键
        sorted_items = sorted(tf.items(), key=lambda x: x[1], reverse=True)

        # 输出前 10 个键
        top_30_keys = [item for item in sorted_items[:40] if item[0] not in chinese_stopwords + english_stopwords]

        # 输出结果
        print(f"Emotion: {emotion}")
        print("前 10 个值最大的键:", top_30_keys)
        print("="*50)


def trend():
    # 时间序列分析
    # 读取数据并插值填充
    data = pd.read_csv('opinion.csv', parse_dates=['date'])
    data.set_index('date', inplace=True)

    # 进行线性插值填充
    data = data.interpolate(method='linear')

    # ADF检验函数
    def adf_test(series, name=''):
        result = adfuller(series)
        print(f"ADF检验结果 - {name}:")
        print(f"  ADF统计量: {result[0]}")
        print(f"  p值: {result[1]}")
        print(f"  临界值: {result[4]}")
        print('-' * 40)

    # 对每列数据进行ADF检验
    for column in data.columns:
        adf_test(data[column], column)

    # 建立VAR模型并自动选择滞后阶数
    model = VAR(data)
    lag_order_results = model.select_order()

    # 输出最佳滞后阶数（根据不同的信息准则）
    print("最佳滞后阶数:")
    print(f"AIC: {lag_order_results.aic}")

    import pandas as pd
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    # 选择滞后阶数
    lag_order = 1  # 可以根据AIC/BIC选择最佳滞后阶数

    # 进行Johansen协整检验
    johansen_test = coint_johansen(data, det_order=0, k_ar_diff=lag_order)

    # 输出协整检验的结果
    print("Johansen协整检验结果:")
    print(f"Eigenvalue: {johansen_test.eig}")
    print(f"Critical Values (95%): {johansen_test.cvt[:, 1]}")
    print(f"Trace Statistic: {johansen_test.lr1}")
    print(f"Critical Values (95% - Trace): {johansen_test.cvm[:, 1]}")

    from statsmodels.tsa.vector_ar.vecm import VECM

    # 使用选定的滞后阶数建立VECM模型
    vecm = VECM(data, k_ar_diff=lag_order)
    vecm_fitted = vecm.fit()

    # 输出模型结果
    print(vecm_fitted.summary())

    # 定义时间窗口
    window = 4

    # 计算滚动协方差
    cov_matrix = data.rolling(window).cov(pairwise=True)

    # 获取协方差矩阵的部分（不包含NaN值）
    cov_values = cov_matrix.dropna()

    # 比较协方差的符号，判断变化趋势是否相同
    # 如果协方差为正，说明序列的变化趋势相同；如果为负，说明变化趋势相反
    same_trend_periods = []

    # 检查每个时间段内三个时间序列的协方差符号
    for i in range(len(cov_values)):
        ds = cov_values.iloc[i]

        if i % 3 == 0:
            cov_tieba_douban = ds['douban']
            cov_tieba_weibo = ds['weibo']
        if i % 3 == 1:
            cov_douban_weibo = ds['weibo']

        if i % 3 == 2:
            # 检查协方差的符号是否相同
            if (cov_tieba_douban > 0.01 and cov_tieba_weibo > 0.01 and cov_douban_weibo > 0.01) or \
            (cov_tieba_douban < -0.01 and cov_tieba_weibo < -0.01 and cov_douban_weibo < -0.01):
                same_trend_periods.append(ds.name[0])

    # 输出具有相同变化趋势的时间段
    print("时间序列变化趋势相同的时间段：")
    for period in same_trend_periods:
        print(period)


