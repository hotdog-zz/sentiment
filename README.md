# sentiment

## 介绍

互联网社交平台已经成为社会热点事件的主要讨论平台，对网民言论进行情感分析和观点分析已成为社会舆论的主要分析方式，帮助研究人员分析事件起因，民众看法和未来走向。大语言模型在这一任务上给出了新的分析方式，然而其存在着无法分辨网络用语，句子级别的分析无法给出整体趋势，以及可解释性差等问题。对此我们提出了一个数据增强策略，并构建了Sentiment-198k用于微调模型。在使用大模型分析爬取数据后，我们以网民情感为主要分析对象，用时间序列分析说明了不同平台间情感的趋势变化及关键词，并进一步使用词频分析给出了细粒度情感特征词；以网民观点为主要分析对象，分析了不同平台观点的时间趋势和观点关键词，并分析了多层评论平台的观点聚集。最后，我们讨论了网络热点事件的主要发言模式，并以语义空间聚类的方式进行验证。

## 仓库结构
以下是我们的仓库结构
```
sentiment/
│
├── data/                   # 用于下载并存放原始数据集
│   └── oc_emotion/         # oc_emotion数据示例
│       └── README.md       # 数据下载和存放流程
│
├── llm_data/               # 模型推理后数据及分析数据
│   ├── emotion.csv         # 情感数据日平均值
│   ├── opinion.csv         # 观点数据日平均值
|
├── src/                    # 各类代码
│   ├── spider/             # 爬虫代码
│   ├── clean_data.py       # 原始数据集清洗
│   ├── llm_data.py         # 原始数据集增强流程
│   ├── llm_inference.py    # 爬虫数据集分类流程
│   ├── llm_train.py        # 模型训练数据构建
│   └── src.py              # 分析代码
|
├── README.md               # Documentation (this file)
└── LICENSE                 # License file
```

## Usage
### Install Dependencies
```bash
# LLaMA-Factory安装
cd LLaMA-Factory
pip3 install -e ".[torch,metrics]"

# vllm安装
pip3 install vllm

# deepseed安装
pip3 install deepspeed==0.15.4

# the newest llama-factory has a bug with transformers, so we need to install a custom transformers version,
# you can see this issus https://github.com/huggingface/transformers/issues/34503#issuecomment-2448933790
pip3 install git+https://github.com/techkang/transformers.git
```

## License:
[Apache2.0 License](https://github.com/hotdog-zz/sentiment/blob/main/LICENSE)
