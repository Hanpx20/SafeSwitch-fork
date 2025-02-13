# import pandas as pd
# import re


# # 读取 Parquet 文件
# df = pd.read_parquet("/shared/nas2/ph16/TinyZero/mmlu_data/test.parquet")
# dict_list = df.to_dict(orient="records")
# print(dict_list[0]["prompt"][0]["content"])


import ray
from ray.util.placement_group import placement_group, remove_placement_group, get_placement_group

ray.init(_temp_dir="/home/ph16/TinyZero/tmp")

# 试图根据名称获取已经存在的 placement group
pg_name = "global_poolverl_group_2:0"
pg = get_placement_group(pg_name)
if pg is not None:
    print(f"找到名称为 {pg_name} 的 placement group，准备删除。")
    remove_placement_group(pg)
else:
    print(f"没有找到名称为 {pg_name} 的 placement group。")

