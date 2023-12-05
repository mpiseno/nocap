import os
import pathlib
import argparse

import numpy as np


DATA_ROOT = 'data/amass_raw'
PROCESSED_DATA_ROOT = 'data/amass_processed'
INVALID_FILES = [
    'neutral_stagei.npz'
]

DESIRED_FRAME_RATE = 30
WINDOW = int(DESIRED_FRAME_RATE / 2)

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2


def process_amass_file(input_path):
    print(f'Processing {input_path}...')

    data = np.load(input_path)
    frame_rate = data['mocap_frame_rate']
    assert frame_rate >= DESIRED_FRAME_RATE

    body_poses = data['pose_body']

    # Ensure poses are sampled at DESIRED_FRAME_RATE Hz
    num_poses = len(body_poses)
    num_poses_at_desired_frame_rate = int(num_poses * (DESIRED_FRAME_RATE / frame_rate))
    desired_idxs = np.linspace(0, num_poses - 1, num_poses_at_desired_frame_rate).astype(int)
    body_poses = body_poses[desired_idxs]

    # TODO: figure out which indices correspond to which joints
    processed_data = []
    for i in range(0, len(body_poses) - WINDOW, int(DESIRED_FRAME_RATE / 10)):
        frames = body_poses[i:i+WINDOW]
        frames = frames.reshape(-1, 3)
        processed_data.append(frames)

    processed_data = np.array(processed_data)
    return processed_data


def process(args):
    valid_file = lambda fp: fp.endswith('.npz') and fp not in INVALID_FILES

    all_data = []
    for dataset in args.datasets:
        dataset_path = os.path.join(DATA_ROOT, dataset)
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if valid_file(file):
                    input_path = os.path.join(root, file)
                    processed_data = process_amass_file(input_path)
                    all_data.append(processed_data)

    all_data = np.vstack(all_data)
    np.random.shuffle(all_data)
    print(f'Total amount of data: {all_data.shape}')

    train_end = int(TRAIN_SPLIT * len(all_data))
    train_split = all_data[:train_end]
    val_split = all_data[train_end:]

    output_root = os.path.join(PROCESSED_DATA_ROOT, '_'.join(args.datasets))
    pathlib.Path(output_root).mkdir(parents=True, exist_ok=True)
    train_path = os.path.join(output_root, 'train_split')
    val_path = os.path.join(output_root, 'val_split')
    train_save_dict = {
        'pose_body': train_split,
        'motion_freq': DESIRED_FRAME_RATE
    }
    val_save_dict = {
        'pose_body': val_split,
        'motion_freq': DESIRED_FRAME_RATE
    }
    np.savez(train_path, **train_save_dict)
    np.savez(val_path, **val_save_dict)


def verify(args):
    for dataset in args.datasets:
        dataset_path = os.path.join(DATA_ROOT, dataset)
        dataset_path_exists = os.path.isdir(dataset_path)
        assert dataset_path_exists, f'Dataset path {dataset_path} does not exist.'
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', type=str, required=True)
    args = parser.parse_args()
    verify(args)
    process(args)