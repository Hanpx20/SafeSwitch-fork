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

import re


def extract_solution(solution_str, method='strict'):
    assert method in ['strict', 'flexible']
    if len(solution_str) == 1:
        return solution_str
    if len(solution_str) == 3:
        return solution_str[1]
    matches = re.findall(r'{(.*?)}', solution_str)
    return matches[-1].strip().upper() if matches else None



def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    # print(solution_str, ground_truth)
    # exit(0)
    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return score
        elif answer in ['A', 'B', 'C', 'D']: # if the answer is one valid choice, then give partial score
            return format_score
        else:
            return 0