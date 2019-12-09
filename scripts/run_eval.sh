EVAL_ROOT=/home/naplings/src/model/lyft_all_fhd/eval_results/step_151200
EVAL_RESULT=${EVAL_ROOT}/results_nusc.json
DATASET_DIR=/home/naplings/Datasets/lyft/train
JSON_DIR=${EVAL_ROOT}
OUTPUT_FILE=${EVAL_ROOT}/eval_result_second

python eval.py eval --second_result_dir=${EVAL_RESULT} --dataset_dir=${DATASET_DIR} --json_output_dir=${JSON_DIR} --output_file_dir=${OUTPUT_FILE} --n_jobs=6
