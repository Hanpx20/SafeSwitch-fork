# TinyZero
![image](cover.png)

TinyZero is a reproduction of [DeepSeek R1 Zero](https://github.com/deepseek-ai/DeepSeek-R1) in countdown and multiplication tasks. We built upon [veRL](https://github.com/volcengine/verl).

Through RL, the 3B base LM develops self-verification and search abilities all on its own 

You can experience the Ahah moment yourself for < $30 

Twitter thread: https://x.com/jiayi_pirate/status/1882839370505621655

Full experiment log: https://wandb.ai/jiayipan/TinyZero

Paper's on it's way!

## Installation

```
conda create -n zero python=3.9
# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1
pip3 install ray

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
# quality of life
pip install wandb IPython matplotlib
```

## Countdown task

**Update: Now We Can Using Tinyzero to train on MMLU**
You can refer to `scripts/training_script.sh` for procedure, and you may have to set some environment variables.


**Data Preparation**
```
conda activate zero
python ./examples/data_preprocess/countdown.py --local_dir {path_to_your_dataset}
python ./examples/data_preprocess/mmlu.py --local_dir {path_to_your_dataset}
```

If using a Qwen-Instruct model, add the parameter: `--template_type=qwen-instruct`

**Training**

```
conda activate zero
```
For the following code, if you see Out-of-vram, try add `critic.model.enable_gradient_checkpointing=True` to the script, and checkout the discussion [here](https://github.com/Jiayi-Pan/TinyZero/issues/5#issuecomment-2624161643)

The code will read the `data_source` column in data file and decide which reward function to use.
```
export CUDA_VISIBLE_DEVICES=0,5
export N_GPUS=2
export BASE_MODEL={path_to_your_model}
export DATA_DIR={path_to_your_dataset}
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME={name}
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero.sh
```

**Evaluation**
For evaluating on MMLU, run:
```
python inference/inference.py \
    --model-path ${model_id} \
    --input-file ${data_file} \
    --run-name ${run_name} \
    --output-dir model_answers \
    --limit 1

python inference/eval_mmlu.py \
    --run-name ${run_name} \
    --output-dir model_answers
```


## Acknowledge
* We run our experiments based on [veRL](https://github.com/volcengine/verl).
* We use Qwen2.5 series base model [Qwen2.5](https://github.com/QwenLM/Qwen2.5).

## Citation
```
@misc{tinyzero,
author       = {Jiayi Pan and Junjie Zhang and Xingyao Wang and Lifan Yuan and Hao Peng and Alane Suhr},
title        = {TinyZero},
howpublished = {https://github.com/Jiayi-Pan/TinyZero},
note         = {Accessed: 2025-01-24},
year         = {2025}
}
```
