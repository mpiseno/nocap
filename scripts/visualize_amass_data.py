import os
import argparse

import torch
import numpy as np

from nocap.utils.torch import copy2cpu as c2c
from nocap.body_model.body_model import BodyModel

# import trimesh
# from body_visualizer.tools.vis_tools import colors
# from body_visualizer.mesh.mesh_viewer import MeshViewer
# from body_visualizer.mesh.sphere import points_to_spheres
# from body_visualizer.tools.vis_tools import show_image


data_root = 'data/amass_raw'
smplh_dir = 'nocap/body_model/smplh'


def vis_body_pose_beta(body_pose_beta, faces, mv, fId=0):
    body_mesh = trimesh.Trimesh(vertices=c2c(body_pose_beta.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False)
    show_image(body_image)


def visualize(args, device):
    data_dir = os.path.join(data_root, args.dataset)
    
    frame = os.path.join(data_dir, '0005/0005_Jogging001_stageii.npz') # the path to body data
    bdata = np.load(frame)

    import pdb; pdb.set_trace()

    print(f'Data keys available: {list(bdata.keys())}')

    subject_gender = bdata['gender']

    bm_fname = os.path.join(smplh_dir, f'{subject_gender}/model.npz')
    num_betas = 16 # number of body parameters
    bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas).to(device)
    faces = c2c(bm.f)

    body_parms = {
        'root_orient': torch.Tensor(bdata['poses'][:, :3]).to(device), # controls the global root orientation
        'pose_body': torch.Tensor(bdata['poses'][:, 3:66]).to(device), # controls the body
        'pose_hand': torch.Tensor(bdata['poses'][:, 66:]).to(device), # controls the finger articulation
        'trans': torch.Tensor(bdata['trans']).to(device), # controls the global body position
        'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=time_length, axis=0)).t0(device), # controls the body shape. Body shape is static
    }

    print('Body parameter vector shapes: \n{}'.format(' \n'.join(['{}: {}'.format(k,v.shape) for k,v in body_parms.items()])))
    time_length = len(bdata['trans'])
    print('time_length = {}'.format(time_length))

    imw, imh = 1600, 1600
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

    # forward pass through body bodel
    body_pose_beta = bm(**{k:v for k,v in body_parms.items() if k in ['pose_body', 'betas']})
    vis_body_pose_beta(body_pose_beta, faces, mv, fId=0)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--num_samples', default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visualize(args, device)


if __name__ == '__main__':
    main()


'''
pose_body: (63,) dimensional joint positions. 3 rotational DoF parameterized by exponential coordinates
    It is unclear which position corresponds to which joint.
'''