# sentiment

## 仓库结构
以下是我们的仓库结构
```
sentiment/
│
├── data/                   # 用于下载并存放原始数据集
│   ├── oc_emotion/         # oc_emotion数据示例
│       ├── README.md       # 数据下载和存放流程
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
│   ├── src.py              # 分析代码
|
├── README.md               # Documentation (this file)
└── LICENSE                 # License file
```
