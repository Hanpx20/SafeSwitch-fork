import pandas as pd
import re


# # 读取 Parquet 文件
df = pd.read_parquet("/shared/nas2/ph16/TinyZero/mmlu_data_qwen/test.parquet")
dict_list = df.to_dict(orient="records")
print(dict_list[0]["prompt"][0]["content"])
print('-'*50)
df = pd.read_parquet("/shared/nas2/ph16/TinyZero/mmlu_data/test.parquet")
dict_list = df.to_dict(orient="records")
print(dict_list[0]["prompt"][0]["content"])

# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("/shared/nas2/shared/llms/Qwen2.5-3B")
# print(tokenizer.chat_template)
