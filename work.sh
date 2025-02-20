export CUDA_VISIBLE_DEVICES=1,2
export N_GPUS=2
export BASE_MODEL=/shared/nas2/shared/llms/Qwen2.5-3B-Instruct # 
# export BASE_MODEL=/shared/nas2/ph16/TinyZero/checkpoints/countdown/Qwen2.5-3B-Intruct-grpo/actor/global_step_1100
export DATA_DIR=/shared/nas2/ph16/TinyZero/countdown_data_qwen #
export OUTPUT_DIR=/shared/nas2/ph16/TinyZero/checkpoints #
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown/Qwen2.5-3B-Instruct-schedule #
export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_DEDUP_LOGS=0
export WANDB_ENTITY=hanpx20
export HYDRA_FULL_ERROR=1
export USE_CHAT_TEMPLATE=false  # seem to have bug, don't set to true
set -e
# statistics for mmlu
# small
# The size of training set: 3565
# The size of testing set: 394
# all
# The size of training set: 99842
# The size of testing set: 1531



bash ./scripts/train_tiny_zero_grpo.sh

# # Evaluate on mmlu
# python inference/inference.py \
#     --model-path ${BASE_MODEL} \
#     --input-file /shared/nas2/ph16/TinyZero/open_writing_data/test.jsonl \
#     --run-name ${EXPERIMENT_NAME} \
#     --output-dir model_answers \
#     --limit 1

# python inference/eval_mmlu.py \
#     --run-name ${EXPERIMENT_NAME} \
#     --output-dir model_answers