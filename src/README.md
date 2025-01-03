# 各类代码文件

## 使用方式

### clean_data.py

调用相应函数即可。

### llm_data.py

```
python src/llm_data.py --actor_name meta-llama/Meta-Llama-3-8B-Instruct --model_name meta-llama/Meta-Llama-3-8B-Instruct --dataset_name go_emotion nlpcc_2013 nlpcc_2014 nlpcc_2018 oc_emotion --temperature 0.7 --sample_num 1
```

### llm_inference.py

```
python src/llm_inference.py --actor_name Qwen/Qwen2.5-7B-Instruct --model_name Qwen/Qwen2.5-7B-Instruct --dataset_name tieba_reply --temperature 0 --sample_num 1
```

### llm_train.py

```
python src/llm_train.py --dataset_name go_emotion nlpcc_2013 nlpcc_2014 nlpcc_2018 oc_emotion --actor_name Qwen/Qwen2.5-7B-Instruct THUDM/glm-4-9b-chat meta-llama/Meta-Llama-3-8B-Instruct
```

### src.py

调用相应函数即可。
