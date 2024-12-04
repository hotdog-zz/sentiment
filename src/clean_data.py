import pandas as pd
import pdb
import json

def go_emotion():
    emotion_categories = {
        "positive": ["amusement", "excitement", "joy", "love", "desire", "optimism", "caring", "pride", "admiration", "gratitude", "relief", "approval"],
        "negative": ["fear", "nervousness", "remorse", "embarrassment", "disappointment", "sadness", "grief", "disgust", "anger", "annoyance", "disapproval"],
        "ambiguous": ["realization", "surprise", "curiosity", "confusion", "neutral"]
    }
    # 读取CSV文件
    dfs = [pd.read_csv('data/go_emotion/goemotions_1.csv'), pd.read_csv('data/go_emotion/goemotions_2.csv'), pd.read_csv('data/go_emotion/goemotions_3.csv')]
    df = pd.concat(dfs, ignore_index=True)

    result = []

    # 遍历数据的每一行
    for idx, row in df.iterrows():
        # 提取text列
        text = row['text']
        
        # 提取情绪列，筛选出值为1的情绪列
        emotions = [emotion for emotion in ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 
                                            'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 
                                            'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 
                                            'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 
                                            'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 
                                            'neutral'] if row[emotion] == 1]
        # 分类情绪
        classify = set()  # 使用集合避免重复分类
        for emotion in emotions:
            if emotion in emotion_categories['positive']:
                classify.add("positive")
            elif emotion in emotion_categories['negative']:
                classify.add("negative")
            elif emotion in emotion_categories['ambiguous']:
                classify.add("ambiguous")
        
        # 将分类结果转换为列表
        classify_list = list(classify)
        # 构建字典并添加到结果列表
        result.append({
            'idx': idx,
            'dataset': 'go_emotion',
            'text': text,
            'classify': classify_list,
            'emotion': emotions
        })
    with open("data/go_emotion/clean_data.json", "w") as f:
        json.dump(result, f, indent=2)
go_emotion()
