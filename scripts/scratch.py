import os

from matplotlib.pyplot import plt
import numpy as np


DESIRED_FRAME_RATE = 30
WINDOW = int(DESIRED_FRAME_RATE)

input_path = 'data/amass_processed/by_file/0005_Jogging001_stageii.npz'

data = np.load(input_path)
frame_rate = data['mocap_frame_rate']
body_poses = data['pose_body']
betas = data['betas']
root_orient = data['root_orient']

# Ensure poses are sampled at DESIRED_FRAME_RATE Hz
num_poses = len(body_poses)
num_poses_at_desired_frame_rate = int(num_poses * (DESIRED_FRAME_RATE / frame_rate))
desired_idxs = np.linspace(0, num_poses - 1, num_poses_at_desired_frame_rate).astype(int)
body_poses = body_poses[desired_idxs]

for i in range(len(body_poses)):
    import pdb; pdb.set_trace()
    jnt = body_poses[i]



# # TODO: figure out which indices correspond to which joints
# processed_data = []
# for i in range(0, len(body_poses) - WINDOW):
#     frames = body_poses[i:i+WINDOW]
#     frames = frames.reshape(-1, 3)
#     frames = concat_joint_encodings(frames)
#     processed_data.append(frames)

# processed_data = np.array(processed_data)