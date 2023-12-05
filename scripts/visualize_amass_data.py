import os
import argparse
import pathlib

from PIL import Image

import trimesh
import torch
import numpy as np
import tqdm

from scipy.spatial.transform import Rotation as R

from nocap.utils.torch import copy2cpu as c2c
from nocap.utils.io import write_video
from body_visualizer.tools.vis_tools import colors
from body_visualizer.mesh.mesh_viewer import MeshViewer
# from body_visualizer.mesh.sphere import points_to_spheres
from human_body_prior.body_model.body_model import BodyModel
from body_visualizer.tools.vis_tools import show_image


smplh_dir = 'nocap/body_model/smplh'
OUTPUT_DIR = 'visuals'
MAX_NUM_FRAMES = 1500


def get_camera_pose():
    # These parameters were found through grad student search
    r = R.from_rotvec([-np.pi/6, 2*np.pi/3, np.pi])
    rot = r.as_matrix()
    pose = np.eye(4)
    pose[:3, 3] = np.array([-1.5, 1.5, 1.])
    pose[:3, :3] = rot
    return pose


def vis_body_pose_beta(body_pose_beta, faces, mv, fId=0):
    body_mesh = trimesh.Trimesh(vertices=c2c(body_pose_beta.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False)
    return body_image


def visualize(args, device):
    bdata = np.load(args.motion_path)

    print(f'Data keys available: {list(bdata.keys())}')

    subject_gender = bdata['gender']

    bm_fname = os.path.join(smplh_dir, f'{subject_gender}/model.npz')
    num_betas = 16 # number of body parameters
    bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, model_type='smplh').to(device)
    faces = c2c(bm.f)

    time_length = len(bdata['trans'])
    print('time_length = {}'.format(time_length))
    body_parms = {
        'root_orient': torch.Tensor(bdata['poses'][:, :3]).to(device), # controls the global root orientation
        'pose_body': torch.Tensor(bdata['poses'][:, 3:66]).to(device), # controls the body
        'pose_hand': torch.Tensor(bdata['poses'][:, 66:]).to(device), # controls the finger articulation
        'trans': torch.Tensor(bdata['trans']).to(device), # controls the global body position
        'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=time_length, axis=0)).to(device), # controls the body shape. Body shape is static
    }

    imw, imh = 512, 512
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
    mv.update_camera_pose(get_camera_pose())

    # forward pass through body bodel
    body_pose_beta = bm(**{k:v for k,v in body_parms.items() if k in ['pose_body', 'betas', 'root_orient']})

    vis_body_pose_beta(body_pose_beta, faces, mv, fId=0)

    print(f'Rendering Video...')
    images = []
    num_frames = min(MAX_NUM_FRAMES, len(body_pose_beta.v))
    for t in tqdm.tqdm(range(num_frames)):
        body_image = vis_body_pose_beta(body_pose_beta, faces, mv, fId=t)
        images.append(body_image)
    
    motion_file = args.motion_path.split('/')[-1]
    output_path = os.path.join(OUTPUT_DIR, motion_file[:-len('.npz')] + '.mp4')
    output_dir = os.path.join(*output_path.split('/')[:-1])
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f'Writing video to {output_path}')
    write_video(output_path, images, fps=60, width=imw, height=imh) # has to be same width height as body_image. no resize here




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--motion_path', required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visualize(args, device)


if __name__ == '__main__':
    main()


'''
pose_body: (63,) dimensional joint positions. 3 rotational DoF parameterized by exponential coordinates
    It is unclear which position corresponds to which joint.
'''
