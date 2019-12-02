import json
import fire
import pickle
from copy import copy
from nuscenes.nuscenes import NuScenes
from pathlib import Path
import subprocess
import multiprocessing

key_mapping = {
    'sample_token': 'sample_token',
    'translation': 'translation',
    'size': 'size',
    'rotation': 'rotation',
    'detection_name': 'name',
    'detection_score': 'score'
}

nusc_key_mappings = {
    'sample_token': 'sample_token',
    'translation': 'translation',
    'size': 'size',
    'rotation': 'rotation',
    'category_name': 'name'
}


def convert_second_results(second_result_dir, dataset_dir, output_dir):
    with open(second_result_dir, 'rb') as f:
        data = json.load(f)

    prediction_template = {
        'sample_token': '',
        'translation': [],
        'size': [],
        'rotation': [],
        'name': '',
        'score': 0
    }

    nusc = NuScenes(version='v1.0-trainval', dataroot=dataset_dir)

    predictions = list()
    ground_truth = list()

    print('Converting reuslts and ground truth ...')
    for token, value in data['results'].items():
        for item in value:
            pred = copy(prediction_template)
            for k, v in key_mapping.items():
                pred[v] = item[k]
            predictions.append(pred)

        anns = nusc.get('sample', token)['anns']
        for ann_token in anns:
            gt = copy(prediction_template)
            ann_meta = nusc.get('sample_annotation', ann_token)
            for k, v in nusc_key_mappings.items():
                gt[v] = ann_meta[k]
            ground_truth.append(gt)

    with open(Path(output_dir) / 'predictions.json', 'w') as f:
        print('Saving Predictions ...')
        json.dump(predictions, f)

    with open(Path(output_dir) / 'ground_truth.json', 'w') as f:
        print('Saving Ground Truth ...')
        json.dump(ground_truth, f)

    print('Done Converting.')


def eval(second_result_dir, dataset_dir, json_output_dir, output_file_dir, n_jobs=6):
    curr_dir = Path(__file__).resolve().parent
    pred_path = Path(json_output_dir)/'predictions.json'
    gt_path = Path(json_output_dir)/'ground_truth.json'

    if not (pred_path.exists() and gt_path.exists()):
        convert_second_results(second_result_dir, dataset_dir, json_output_dir)

    cmds = [['python',
             str(curr_dir/'mAP_evaluation.py'),
             '--pred_file', str(pred_path),
             '--gt_file', str(gt_path),
             '--iou_threshold', str(i),
             '-o', str(output_file_dir)]
            for i in [0.01 * x for x in range(50, 100, 5)]]

    with multiprocessing.Pool(n_jobs) as pool:
        pool.map(subprocess.run, cmds)


if __name__ == "__main__":
    fire.Fire()
