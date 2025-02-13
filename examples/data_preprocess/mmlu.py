# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import re
import os
import datasets
import pandas as pd
from verl.utils.hdfs_io import copy, makedirs
import argparse


def parquet_to_json(parquet_path, json_path, orient='records', lines=True, encoding='utf-8'):
    """d
        parquet_path: Parquet 文件路径
        json_path: 输出 JSON 文件路径
        orient: JSON 的格式方向，默认为 'records'
        lines: 是否每行一个 JSON 对象，默认为 True（即 NDJSON 格式）
        encoding: 写入文件时的编码，默认为 'utf-8'
    """
    df = pd.read_parquet(parquet_path)
    json_str = df.to_json(orient=orient, lines=lines, force_ascii=False)
    # 写入 JSON 文件
    with open(json_path, 'w', encoding=encoding) as f:
        f.write(json_str)



id_to_alpha = ['A', 'B', 'C', 'D']
selected_categories = ['business_ethics', 'computer_security', 'formal_logic', 'high_school_government_and_politics', 'jurisprudence', 'logical_fallacies',  'moral_disputes', 'moral_scenarios', 'professional_law']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/mmlu')
    parser.add_argument('--all', action='store_true')

    args = parser.parse_args()
    if args.all:
        args.local_dir += '_all'

    num_few_shot = 5
    data_source = "cais/mmlu"

    '''
    'auxiliary_train' is only provided for 'all'
    other categories only have 'valid' and 'test'
    since 'test' is bigger, we use 'test' for training and 'valid' for testing
    '''
    if args.all:
        dataset = datasets.load_dataset(data_source, "all")
        train_dataset = dataset['auxiliary_train']
        test_dataset = dataset['validation']
    else:
        dataset_list = [datasets.load_dataset(data_source, x) for x in selected_categories]
        train_dataset = datasets.concatenate_datasets([dataset["test"] for dataset in dataset_list])
        test_dataset = datasets.concatenate_datasets([dataset["validation"] for dataset in dataset_list])

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            example['choices'] = ['(' + id_to_alpha[idx] + ') ' + choice for idx, choice in enumerate(example['choices'])]
            
            question = "Question: " + example.pop('question') + '\nChoices:\n' + '\n'.join(example['choices']) + "\nPlease express your thought step by step. Give your final answer, a single letter, between '(' and ')'."

            answer_raw = example.pop('answer')
            solution = id_to_alpha[answer_raw]
            data = {
                "data_source": data_source,
                "subject": example['subject'],
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': solution,
                    "question": question,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    print(f"The size of training set: {len(train_dataset)}")
    print(f"The size of testing set: {len(test_dataset)}")
    local_dir = args.local_dir
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    parquet_to_json(os.path.join(local_dir, 'train.parquet'), os.path.join(local_dir, 'train.jsonl'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    parquet_to_json(os.path.join(local_dir, 'test.parquet'), os.path.join(local_dir, 'test.jsonl'))
