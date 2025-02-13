export CUDA_VISIBLE_DEVICES=4
export model_name=Qwen2.5-1.5B-Instruct
export BASE_MODEL=/shared/nas2/ph16/TinyZero/checkpoints/finetune/${model_name}/final_model
export EXPERIMENT_NAME=mmlu-${model_name}-finetune


bash scripts/eval_mmlu.sh $BASE_MODEL $EXPERIMENT_NAME 0




# The size of training set: 3565
# The size of testing set: 394
# The size of training set: 99842
# The size of testing set: 1531


# for step in {100..1500..100}
# do
#     export BASE_MODEL=/shared/nas2/ph16/TinyZero/checkpoints/mmlu-${model_name}/actor/global_step_$step
#     export EXPERIMENT_NAME=mmlu-${model_name}-rl$step
#     bash scripts/eval_mmlu.sh $BASE_MODEL $EXPERIMENT_NAME 0
# done
