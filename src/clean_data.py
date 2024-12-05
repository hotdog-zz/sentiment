from pydoc import classify_class_attrs
import xml.etree.ElementTree as ET
import pandas as pd
import pdb
import json


def generate_classify(emotions):
    emotion_categories = {
        "positive": ["amusement", "excitement", "joy", "love", "desire", "optimism", "caring", "pride", "admiration", "gratitude", "relief", "approval"],
        "negative": ["fear", "nervousness", "remorse", "embarrassment", "disappointment", "sadness", "grief", "disgust", "anger", "annoyance", "disapproval"],
        "ambiguous": ["realization", "surprise", "curiosity", "confusion", "neutral"]
    }
    if len(emotions) == 0:
        emotions = ['neutral']
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
    return emotions, classify_list

def go_emotion():
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
        emotions, classify_list = generate_classify(emotions)
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

def oc_emotion():
    df = pd.read_csv('data/oc_emotion/OCEMOTION.csv', sep='\t', header=None)
    result = []
    value_map = {
        "sadness": "sadness",
        "happiness": "joy",
        "disgust": "disgust",
        "anger": "anger",
        "like": "love",
        "surprise": "surprise",
        "fear": "fear"
    }
    # 遍历数据的每一行
    for idx, row in df.iterrows():
        # 提取text列
        text = row[1]
        row[2] = value_map.get(row[2], row[2])        
        emotions = [row[2]]
        emotions, classify_list = generate_classify(emotions)

        # 构建字典并添加到结果列表
        result.append({
            'idx': idx,
            'dataset': 'oc_emotion',
            'text': text,
            'classify': classify_list,
            'emotion': emotions
        })
    with open("data/oc_emotion/clean_data.json", "w") as f:
        json.dump(result, f, indent=2)

def nlpcc_2013():
    # 解析XML文件
    tree = ET.parse('data/nlpcc_2013/微博情绪标注语料.xml')  # 替换为你的XML文件路径
    value_map = {
        '愤怒': 'anger',
        '厌恶': 'disgust',
        '恐惧': 'fear',
        '高兴': 'joy',
        '喜好': 'love',
        '悲伤': 'sadness',
        '惊讶': 'surprise'
    }

    result = []
    idx = 0

    root = tree.getroot()  # 获取根元素
    for para in root:
        for sentence in para:
            text = sentence.text
            attrib = sentence.attrib
            emotions = []
            if attrib['opinionated'] == 'Y':
                emotions = [attrib['emotion-1-type'], attrib['emotion-2-type']]
                emotions = [value_map.get(emotion, emotion) for emotion in emotions if emotion in value_map]
            emotions, classify_list = generate_classify(emotions)

            # 构建字典并添加到结果列表
            result.append({
                'idx': idx,
                'dataset': 'nlpcc_2013',
                'text': text,
                'classify': classify_list,
                'emotion': emotions
            })
            idx += 1
    tree = ET.parse('data/nlpcc_2013/微博情绪样例数据V5-13.xml')  # 替换为你的XML文件路径
    root = tree.getroot()  # 获取根元素
    for para in root:
        for sentence in para:
            text = sentence.text
            attrib = sentence.attrib
            emotions = []
            if attrib['emotion_tag'] == 'Y':
                emotions = [attrib['emotion-1-type'], attrib['emotion-2-type']]
                emotions = [value_map.get(emotion, emotion) for emotion in emotions if emotion in value_map]
            emotions, classify_list = generate_classify(emotions)

            # 构建字典并添加到结果列表
            result.append({
                'idx': idx,
                'dataset': 'nlpcc_2013',
                'text': text,
                'classify': classify_list,
                'emotion': emotions
            })
            idx += 1
    with open("data/nlpcc_2013/clean_data.json", "w") as f:
        json.dump(result, f, indent=2)

def nlpcc_2014():
    # 解析XML文件
    tree = ET.parse('data/nlpcc_2014/NLPCC2014微博情绪分析样例数据.xml')  # 替换为你的XML文件路径
    value_map = {
        "sadness": "sadness",
        "happiness": "joy",
        "disgust": "disgust",
        "anger": "anger",
        "like": "love",
        "surprise": "surprise",
        "fear": "fear"
    }

    result = []
    idx = 0

    root = tree.getroot()  # 获取根元素
    for para in root:
        full_text= ""
        for sentence in para:
            text = sentence.text
            full_text += text + " "
            attrib = sentence.attrib
            emotions = []
            if attrib['opinionated'] == 'Y':
                emotions = [attrib['emotion-1-type'], attrib['emotion-2-type']]
                emotions = [value_map.get(emotion, emotion) for emotion in emotions if emotion in value_map]
            emotions, classify_list = generate_classify(emotions)

            # 构建字典并添加到结果列表
            result.append({
                'idx': idx,
                'dataset': 'nlpcc_2013',
                'text': text,
                'classify': classify_list,
                'emotion': emotions
            })
            idx += 1
        attrib = para.attrib
        emotions = [attrib['emotion-type1'], attrib['emotion-type2']]
        emotions = [value_map.get(emotion, emotion) for emotion in emotions if emotion in value_map]
        emotions, classify_list = generate_classify(emotions)

        # 构建字典并添加到结果列表
        result.append({
            'idx': idx,
            'dataset': 'nlpcc_2014',
            'text': full_text,
            'classify': classify_list,
            'emotion': emotions
        })
        idx += 1
    
    tree = ET.parse('data/nlpcc_2014/EmotionClassficationTest.xml')  # 替换为你的XML文件路径
    root = tree.getroot()  # 获取根元素
    for para in root:
        full_text= ""
        for sentence in para:
            text = sentence.text
            full_text += text + " "
            attrib = sentence.attrib
            emotions = []
            if attrib['opinionated'] == 'Y':
                emotions = [attrib['emotion-1-type'], attrib['emotion-2-type']]
                emotions = [value_map.get(emotion, emotion) for emotion in emotions if emotion in value_map]
            emotions, classify_list = generate_classify(emotions)

            # 构建字典并添加到结果列表
            result.append({
                'idx': idx,
                'dataset': 'nlpcc_2013',
                'text': text,
                'classify': classify_list,
                'emotion': emotions
            })
            idx += 1
        attrib = para.attrib
        emotions = [attrib['emotion-type1'], attrib['emotion-type2']]
        emotions = [value_map.get(emotion, emotion) for emotion in emotions if emotion in value_map]
        emotions, classify_list = generate_classify(emotions)

        # 构建字典并添加到结果列表
        result.append({
            'idx': idx,
            'dataset': 'nlpcc_2014',
            'text': full_text,
            'classify': classify_list,
            'emotion': emotions
        })
        idx += 1
    
    tree = ET.parse('data/nlpcc_2014/ExpressionTest.xml')  # 替换为你的XML文件路径
    root = tree.getroot()  # 获取根元素
    for para in root:
        full_text= ""
        for sentence in para:
            text = sentence.text
            full_text += text + " "
            attrib = sentence.attrib
            emotions = []
            if attrib['opinionated'] == 'Y':
                emotions = [attrib['emotion-1-type'], attrib['emotion-2-type']]
                emotions = [value_map.get(emotion, emotion) for emotion in emotions if emotion in value_map]
            emotions, classify_list = generate_classify(emotions)

            # 构建字典并添加到结果列表
            result.append({
                'idx': idx,
                'dataset': 'nlpcc_2013',
                'text': text,
                'classify': classify_list,
                'emotion': emotions
            })
            idx += 1
        attrib = para.attrib
        emotions = [attrib['emotion-type1'], attrib['emotion-type2']]
        emotions = [value_map.get(emotion, emotion) for emotion in emotions if emotion in value_map]
        emotions, classify_list = generate_classify(emotions)

        # 构建字典并添加到结果列表
        result.append({
            'idx': idx,
            'dataset': 'nlpcc_2014',
            'text': full_text,
            'classify': classify_list,
            'emotion': emotions
        })
        idx += 1
    
    with open("data/nlpcc_2014/clean_data.json", "w") as f:
        json.dump(result, f, indent=2)

def dair_ai_emotion():
    df = pd.read_parquet('data/dair_ai_emotion/unsplit/train-00000-of-00001.parquet')
    result = []
    map = {0:"sadness",1:"joy", 2:"love", 3:"anger", 4:"fear", 5:"surprise"}
    for idx, row in df.iterrows():
        emotion = map[row['label']]
        _, classify = generate_classify([emotion])
        item = {
            'idx': idx,
            'dataset': 'dair_ai_emotion',
            'text': row['text'],
            'classify': classify,
            'emotion': [emotion]
        }
        result.append(item)
    with open("data/dair_ai_emotion/clean_data.json", "w") as f:
        json.dump(result, f, indent=2)

def smp_2020():
    df1 = pd.read_json('data/SMP/train/usual_train.txt')
    df2 = pd.read_json('data/SMP/train/virus_train.txt')
    df3 = pd.read_json('data/SMP/eval（刷榜数据集）/usual_eval_labeled.txt')
    df4 = pd.read_json('data/SMP/eval（刷榜数据集）/virus_eval_labeled.txt')
    df5 = pd.read_json('data/SMP/test（最终评测集）/真实评测集/usual_test_labeled.txt')
    df6 = pd.read_json('data/SMP/test（最终评测集）/真实评测集/virus_test_labeled.txt')
    df = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)
    result = []
    map = {"sad":"sadness","happy":"joy", "angry":"anger", "surprise":"surprise", "fear":"fear", "neutral":"neutral"}
    for idx, row in df.iterrows():
        emotion = map[row['label']]
        _, classify = generate_classify([emotion])
        item = {
            'idx': idx,
            'dataset': 'smp_2020',
            'text': row['content'],
            'classify': classify,
            'emotion': [emotion]
        }
        result.append(item)
    with open("data/SMP/clean_data.json", "w") as f:
        json.dump(result, f, indent=2)

def yf_dianping():
    df = pd.read_csv('data/yf_dianping/ratings.csv')
    result = []

    # 遍历数据的每一行
    idx = 0
    for row_num, row in df.iterrows():
        # 提取text列
        text = row['comment']
        rating = row['rating']
        if pd.isna(text):
            continue
        if pd.isna(rating):
            if not pd.isna(row['rating_env']) and not pd.isna(row['rating_flavor']) and not pd.isna(row['rating_service']):
                rating = (row['rating_env'] + row['rating_flavor'] + row['rating_service']) / 3
            else:
                continue
        classify_list = []
        if rating >= 4.0:
            classify_list = ['positive']
        elif rating <= 2.0:
            classify_list = ['negative']
        else:
            classify_list = ['ambiguous']
        emotions = []

        # 构建字典并添加到结果列表
        result.append({
            'idx': idx,
            'dataset': 'yf_dianping',
            'text': text,
            'classify': classify_list,
            'emotion': emotions
        })
        idx += 1
    with open("data/yf_dianping/clean_data.json", "w") as f:
        json.dump(result, f, indent=2)

def online_shopping_10():
    df = pd.read_csv('data/online_shopping_10/online_shopping_10_cats.csv')
    result = []
    # 遍历数据的每一行
    for idx, row in df.iterrows():
        # 提取text列
        text = row['review']    
        label = row['label']
        if label == 1:
            classify_list = ['positive']
        if label == 0:
            classify_list = ['negative']
        emotions = []
        # 构建字典并添加到结果列表
        result.append({
            'idx': idx,
            'dataset': 'online_shopping_10',
            'text': text,
            'classify': classify_list,
            'emotion': emotions
        })
    with open("data/online_shopping_10/clean_data.json", "w") as f:
        json.dump(result, f, indent=2)
