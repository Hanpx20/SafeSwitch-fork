export CUDA_VISIBLE_DEVICES=4
export model_name=Qwen2.5-1.5B-Instruct
export BASE_MODEL=/shared/nas2/ph16/TinyZero/checkpoints/finetune/${model_name}/final_model
export EXPERIMENT_NAME=mmlu-${model_name}-finetune


bash scripts/eval_mmlu.sh $BASE_MODEL $EXPERIMENT_NAME 0