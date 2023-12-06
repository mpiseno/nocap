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
WINDOW = int(DESIRED_FRAME_RATE)

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2

N_JOINTS = 21


def concat_joint_encodings(frames):
    n_frames = int(frames.shape[0] / N_JOINTS)
    joint_encodings = np.eye(N_JOINTS)
    joint_encodings = np.tile(joint_encodings, (n_frames, 1))
    frames = np.concatenate((frames, joint_encodings), axis=-1)
    return frames


def process_amass_file(input_path):
    print(f'Processing {input_path}...')

    data = np.load(input_path)
    frame_rate = data['mocap_frame_rate']
    assert frame_rate >= DESIRED_FRAME_RATE

    body_poses = data['pose_body']
    betas = data['betas']
    root_orient = data['root_orient']

    # Ensure poses are sampled at DESIRED_FRAME_RATE Hz
    num_poses = len(body_poses)
    num_poses_at_desired_frame_rate = int(num_poses * (DESIRED_FRAME_RATE / frame_rate))
    desired_idxs = np.linspace(0, num_poses - 1, num_poses_at_desired_frame_rate).astype(int)
    body_poses = body_poses[desired_idxs]

    # TODO: figure out which indices correspond to which joints
    processed_data = []
    for i in range(0, len(body_poses) - WINDOW):
        frames = body_poses[i:i+WINDOW]
        frames = frames.reshape(-1, 3)
        frames = concat_joint_encodings(frames)
        processed_data.append(frames)

    processed_data = np.array(processed_data)
    return processed_data, betas, root_orient


def save_by_file(file, data, betas, root_orient):
    output_path = os.path.join(PROCESSED_DATA_ROOT, 'by_file', file)
    save_dict = {
        'pose_body': data,
        'motion_freq': DESIRED_FRAME_RATE,
        'betas': betas,
        'root_orient': root_orient
    }
    np.savez(output_path, **save_dict)


def save_by_dataset(args, all_data):
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


def process(args):
    valid_file = lambda fp: fp.endswith('.npz') and fp not in INVALID_FILES

    if args.by_file:
        output_dir = os.path.join(PROCESSED_DATA_ROOT, 'by_file')
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_data = []
    for dataset in args.datasets:
        dataset_path = os.path.join(DATA_ROOT, dataset)
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if valid_file(file):
                    input_path = os.path.join(root, file)
                    processed_data, betas, root_orient = process_amass_file(input_path)
                    all_data.append(processed_data)
                    if args.by_file:
                        save_by_file(file, processed_data, betas, root_orient)

    if not args.by_file:
        all_data = np.vstack(all_data)
        np.random.shuffle(all_data)
        print(f'Total amount of data: {all_data.shape}')

        save_by_dataset(args, all_data)


def verify(args):
    for dataset in args.datasets:
        dataset_path = os.path.join(DATA_ROOT, dataset)
        dataset_path_exists = os.path.isdir(dataset_path)
        assert dataset_path_exists, f'Dataset path {dataset_path} does not exist.'
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', type=str, required=True)
    parser.add_argument('--by_file', action='store_true', default=False)
    args = parser.parse_args()
    verify(args)
    process(args)