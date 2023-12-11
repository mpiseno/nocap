import os

from matplotlib import pyplot as plt
import numpy as np

from nocap.dataset import NoCAP_DS



DESIRED_FRAME_RATE = 30
WINDOW = int(DESIRED_FRAME_RATE)

input_path = 'data/amass_raw/SFU/0005/0005_Jogging001_stageii.npz'

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

import pdb; pdb.set_trace()


# jntx = []
# jnty = []
# for i in range(len(body_poses)):
#     jnt = body_poses[i].reshape(-1, 3)
#     jntx.append(jnt[0, 0])
#     jnty.append(jnt[0, 1])


# data_recon = 'logs/0005_Jogging_overfit_larger_context/samples/seconds=3_ckpt_epoch=final_large_ctx.npz'
# data_recon = np.load(data_recon)

# body_poses_recon = data_recon['pose_body']
# jntx_recon = []
# jnty_recon = []
# for i in range(len(body_poses_recon)):
#     jnt = body_poses_recon[i].reshape(-1, 3)
#     jntx_recon.append(jnt[0, 0])
#     jnty_recon.append(jnt[0, 1])


# # Create a figure and a 3D axis
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# cutoff = 50
# print(cutoff)
# t = np.arange(cutoff)


# # Plot the 3D curve
# ax.plot(t, jntx[:cutoff], jnty[:cutoff], label='Ground Truth')
# ax.plot(t, jntx_recon[:cutoff], jnty_recon[:cutoff], label='Reconstruction')

# # Customize the plot
# ax.set_xlabel('Time (30Hz)')
# ax.set_ylabel('Roll Axis')
# ax.set_zlabel('Pitch Axis')
# ax.legend()

# ax.set_xticks([0, 25, 50])
# ax.set_yticks([-0.4, -0.1, 0.2])
# ax.set_zticks([-0.15, -0.075, 0])

# plt.savefig('3d.png')

