if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <output_directory> <dataset_path> <no_cot?>"
    exit 1
fi

model_id=$1
run_name=$2
no_cot=$3

set -e


python inference/inference.py \
    --model-path ${model_id} \
    --input-file /shared/nas2/ph16/TinyZero/mmlu_data/test.parquet \
    --run-name ${run_name} \
    --output-dir model_answers \
    --limit 1 \
    --no_cot ${no_cot}

python inference/eval_mmlu.py \
    --run-name ${run_name} \
    --output-dir model_answers