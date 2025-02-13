export CUDA_VISIBLE_DEVICES=1,6
export N_GPUS=2
export BASE_MODEL=/shared/nas2/shared/llms/Qwen2.5-1.5B-Instruct
export DATA_DIR=/shared/nas2/ph16/TinyZero/countdown_data
export OUTPUT_DIR=/shared/nas2/ph16/TinyZero/checkpoints
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-Qwen2.5-1.5B-Instruct1
export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_DEDUP_LOGS=0
export WANDB_ENTITY=hanpx20
export HYDRA_FULL_ERROR=1

# The size of training set: 3565
# The size of testing set: 394
# The size of training set: 99842
# The size of testing set: 1531


# python ./examples/data_preprocess/mmlu.py --local_dir /shared/nas2/ph16/TinyZero/mmlu_data
# python ./examples/data_preprocess/mmlu.py --local_dir /shared/nas2/ph16/TinyZero/mmlu_data --all


# bash scripts/eval_mmlu.sh $BASE_MODEL $EXPERIMENT_NAME

bash ./scripts/train_tiny_zero.sh