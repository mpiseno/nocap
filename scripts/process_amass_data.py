import os
import pathlib

import numpy as np


DATA_DIR = 'data/amass_raw'
PROCESSED_DATA_DIR = 'data/amass_processed'

DESIRED_FRAME_RATE = 30
WINDOW = int(DESIRED_FRAME_RATE / 2)


def process_amass_file(input_path):
    data = np.load(input_path)
    frame_rate = data['mocap_frame_rate']
    assert frame_rate >= 30

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


def process():
    all_data = []
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            #if file.endswith('.npz'):
            if '0005_Jogging' in file:
                input_path = os.path.join(root, file)
                processed_data = process_amass_file(input_path)
                all_data.append(processed_data)

    all_data = np.vstack(all_data)

    # Do train test val split
    train_split = all_data
    train_path = os.path.join(PROCESSED_DATA_DIR, 'train_split.npy')
    np.save(train_path, train_split)
                

if __name__ == '__main__':
    process()