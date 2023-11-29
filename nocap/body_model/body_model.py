import sys
import pickle

import numpy as np
import torch
import torch.nn as nn


from smplx import SMPL, SMPLH, SMPLX
from smplx.vertex_ids import vertex_ids
from smplx.utils import Struct

from nocap.utils.torch import copy2cpu as c2c


class BodyModel(nn.Module):
    '''
    Wrapper around SMPLX body model class.
    '''

    def __init__(self,
                 bm_path,
                 num_betas=10,
                 batch_size=1,
                 num_expressions=10,
                 use_vtx_selector=False,
                 model_type='smplh'):
        super(BodyModel, self).__init__()
        '''
        Creates the body model object at the given path.

        :param bm_path: path to the body model pkl file
        :param num_expressions: only for smplx
        :param model_type: one of [smpl, smplh, smplx]
        :param use_vtx_selector: if true, returns additional vertices as joints that correspond to OpenPose joints
        '''
        self.use_vtx_selector = use_vtx_selector
        cur_vertex_ids = None
        if self.use_vtx_selector:
            cur_vertex_ids = vertex_ids[model_type]
        data_struct = None
        if '.npz' in bm_path:
            # smplx does not support .npz by default, so have to load in manually
            smpl_dict = np.load(bm_path, encoding='latin1')
            data_struct = Struct(**smpl_dict)
            # print(smpl_dict.files)
            if model_type == 'smplh':
                data_struct.hands_componentsl = np.zeros((0))
                data_struct.hands_componentsr = np.zeros((0))
                data_struct.hands_meanl = np.zeros((15 * 3))
                data_struct.hands_meanr = np.zeros((15 * 3))
                V, D, B = data_struct.shapedirs.shape
                data_struct.shapedirs = np.concatenate([data_struct.shapedirs, np.zeros((V, D, SMPL.SHAPE_SPACE_DIM-B))], axis=-1) # super hacky way to let smplh use 16-size beta
        kwargs = {
                'model_type' : model_type,
                'data_struct' : data_struct,
                'num_betas': num_betas,
                'batch_size' : batch_size,
                'num_expression_coeffs' : num_expressions,
                'vertex_ids' : cur_vertex_ids,
                'use_pca' : False,
                'flat_hand_mean' : True
        }
        assert(model_type in ['smpl', 'smplh', 'smplx'])
        if model_type == 'smpl':
            self.bm = SMPL(bm_path, **kwargs)
            self.num_joints = SMPL.NUM_JOINTS
        elif model_type == 'smplh':
            self.bm = SMPLH(bm_path, **kwargs)
            self.num_joints = SMPLH.NUM_JOINTS
        elif model_type == 'smplx':
            self.bm = SMPLX(bm_path, **kwargs)
            self.num_joints = SMPLX.NUM_JOINTS

        self.model_type = model_type

    @property
    def f(self):
        return self.bm.faces_tensor

    def forward(self, root_orient=None, pose_body=None, pose_hand=None, pose_jaw=None, pose_eye=None, betas=None,
                trans=None, dmpls=None, expression=None, return_dict=False, **kwargs):
        '''
        Note dmpls are not supported.
        '''
        assert(dmpls is None)
        out_obj = self.bm(
                betas=betas,
                global_orient=root_orient,
                body_pose=pose_body,
                left_hand_pose=None if pose_hand is None else pose_hand[:,:(SMPLH.NUM_HAND_JOINTS*3)],
                right_hand_pose=None if pose_hand is None else pose_hand[:,(SMPLH.NUM_HAND_JOINTS*3):],
                transl=trans,
                expression=expression,
                jaw_pose=pose_jaw,
                leye_pose=None if pose_eye is None else pose_eye[:,:3],
                reye_pose=None if pose_eye is None else pose_eye[:,3:],
                return_full_pose=True,
                **kwargs
        )

        out = {
            'v' : out_obj.vertices,
            'f' : self.bm.faces_tensor,
            'betas' : out_obj.betas,
            'Jtr' : out_obj.joints,
            'pose_body' : out_obj.body_pose,
            'full_pose' : out_obj.full_pose
        }
        if self.model_type in ['smplh', 'smplx']:
            out['pose_hand'] = torch.cat([out_obj.left_hand_pose, out_obj.right_hand_pose], dim=-1)
        if self.model_type == 'smplx':
            out['pose_jaw'] = out_obj.jaw_pose
            out['pose_eye'] = pose_eye
        

        if not self.use_vtx_selector:
            # don't need extra joints
            out['Jtr'] = out['Jtr'][:,:self.num_joints+1] # add one for the root

        if not return_dict:
            out = Struct(**out)

        return out
    

# class BodyModel(nn.Module):

#     def __init__(self,
#                  bm_fname,
#                  num_betas=10,
#                  num_dmpls=None, dmpl_fname=None,
#                  num_expressions=None,
#                  use_posedirs=True,
#                  model_type=None,
#                  dtype=torch.float32,
#                  persistant_buffer=False):

#         super(BodyModel, self).__init__()

#         '''
#         :param bm_fname: path to a SMPL model as pkl file
#         :param num_betas: number of shape parameters to include.
#         :param device: default on gpu
#         :param dtype: float precision of the computations
#         :return: verts, trans, pose, betas 
#         '''

#         self.dtype = dtype


#         # -- Load SMPL params --
#         if bm_fname.endswith('.npz'):
#             smpl_dict = np.load(bm_fname, encoding='latin1')
#         else:
#             raise ValueError(f'bm_fname must be a .npz file: {bm_fname}')

#         # these are supposed for later convenient look up
#         self.num_betas = num_betas
#         self.num_dmpls = num_dmpls
#         self.num_expressions = num_expressions

#         npose_params = smpl_dict['posedirs'].shape[2] // 3
#         if model_type:
#             self.model_type = model_type
#         else:
#             self.model_type = {12:'flame', 69: 'smpl', 153: 'smplh', 162: 'smplx', 45: 'mano',
#                            105: 'animal_horse', 102: 'animal_dog'}[npose_params]

#         assert self.model_type in ['smpl', 'smplh', 'smplx', 'mano', 'mano', 'animal_horse', 'animal_dog', 'flame', 'animal_rat'], ValueError(
#             'model_type should be in smpl/smplh/smplx/mano.')

#         self.use_dmpl = False
#         if num_dmpls is not None:
#             if dmpl_fname is not None:
#                 self.use_dmpl = True
#             else:
#                 raise (ValueError('dmpl_fname should be provided when using dmpls!'))

#         if self.use_dmpl and self.model_type in ['smplx', 'mano', 'animal_horse', 'animal_dog']: raise (
#             NotImplementedError('DMPLs only work with SMPL/SMPLH models for now.'))

#         self.use_expression = self.model_type in ['smplx','flame'] and num_expressions is not None

#         # Mean template vertices
#         self.comp_register('init_v_template', torch.tensor(smpl_dict['v_template'][None], dtype=dtype), persistent=persistant_buffer)

#         self.comp_register('f', torch.tensor(smpl_dict['f'].astype(np.int32), dtype=torch.int32), persistent=persistant_buffer)

#         num_total_betas = smpl_dict['shapedirs'].shape[-1]
#         if num_betas < 1:
#             num_betas = num_total_betas

#         shapedirs = smpl_dict['shapedirs'][:, :, :num_betas]
#         self.comp_register('shapedirs', torch.tensor(shapedirs, dtype=dtype), persistent=persistant_buffer)

#         if self.use_expression:
#             if smpl_dict['shapedirs'].shape[-1] > 300:
#                 begin_shape_id = 300
#             else:
#                 begin_shape_id = 10
#                 num_expressions = smpl_dict['shapedirs'].shape[-1] - 10

#             exprdirs = smpl_dict['shapedirs'][:, :, begin_shape_id:(begin_shape_id + num_expressions)]
#             self.comp_register('exprdirs', torch.tensor(exprdirs, dtype=dtype), persistent=persistant_buffer)

#             expression = torch.tensor(np.zeros((1, num_expressions)), dtype=dtype)
#             self.comp_register('init_expression', expression, persistent=persistant_buffer)

#         if self.use_dmpl:
#             dmpldirs = np.load(dmpl_fname)['eigvec']

#             dmpldirs = dmpldirs[:, :, :num_dmpls]
#             self.comp_register('dmpldirs', torch.tensor(dmpldirs, dtype=dtype), persistent=persistant_buffer)

#         # Regressor for joint locations given shape - 6890 x 24
#         self.comp_register('J_regressor', torch.tensor(smpl_dict['J_regressor'], dtype=dtype), persistent=persistant_buffer)

#         # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*30 x 207
#         if use_posedirs:
#             posedirs = smpl_dict['posedirs']
#             # print(self.model_type, posedirs.shape)
#             posedirs = posedirs.reshape([posedirs.shape[0] * 3, -1]).T
#             self.comp_register('posedirs', torch.tensor(posedirs, dtype=dtype), persistent=persistant_buffer)
#         else:
#             self.posedirs = None

#         # indices of parents for each joints
#         kintree_table = smpl_dict['kintree_table'].astype(np.int32)
#         self.comp_register('kintree_table', torch.tensor(kintree_table, dtype=torch.int32), persistent=persistant_buffer)

#         # LBS weights
#         # weights = np.repeat(smpl_dict['weights'][np.newaxis], batch_size, axis=0)
#         weights = smpl_dict['weights']
#         self.comp_register('weights', torch.tensor(weights, dtype=dtype), persistent=persistant_buffer)

#         self.comp_register('init_trans', torch.zeros((1,3), dtype=dtype), persistent=persistant_buffer)
#         # self.register_parameter('trans', nn.Parameter(trans, requires_grad=True))

#         # root_orient
#         # if self.model_type in ['smpl', 'smplh']:
#         self.comp_register('init_root_orient', torch.zeros((1,3), dtype=dtype), persistent=persistant_buffer)

#         # pose_body
#         if self.model_type in ['smpl', 'smplh', 'smplx']:
#             self.comp_register('init_pose_body', torch.zeros((1,63), dtype=dtype), persistent=persistant_buffer)
#         elif self.model_type == 'animal_horse':
#             self.comp_register('init_pose_body', torch.zeros((1,105), dtype=dtype), persistent=persistant_buffer)
#         elif self.model_type == 'flame':
#             self.comp_register('init_pose_body', torch.zeros((1,3), dtype=dtype), persistent=persistant_buffer)
#         elif self.model_type in ['animal_dog','animal_rat']:
#             self.comp_register('init_pose_body', torch.zeros((1,102), dtype=dtype), persistent=persistant_buffer)

#         # pose_hand
#         if self.model_type in ['smpl']:
#             self.comp_register('init_pose_hand', torch.zeros((1,1*3*2), dtype=dtype), persistent=persistant_buffer)
#         elif self.model_type in ['smplh', 'smplx']:
#             self.comp_register('init_pose_hand', torch.zeros((1,15*3*2), dtype=dtype), persistent=persistant_buffer)
#         elif self.model_type in ['mano']:
#             self.comp_register('init_pose_hand', torch.zeros((1,15*3), dtype=dtype), persistent=persistant_buffer)

#         # face poses
#         if self.model_type in ['smplx','flame']:
#             self.comp_register('init_pose_jaw', torch.zeros((1,1*3), dtype=dtype), persistent=persistant_buffer)
#             self.comp_register('init_pose_eye', torch.zeros((1,2*3), dtype=dtype), persistent=persistant_buffer)

#         self.comp_register('init_betas', torch.zeros((1,num_betas), dtype=dtype), persistent=persistant_buffer)

#         if self.use_dmpl:
#             self.comp_register('init_dmpls', torch.zeros((1,num_dmpls), dtype=dtype), persistent=persistant_buffer)

#     def comp_register(self, name, value, persistent=False):
#         if sys.version_info[0] > 2:
#             self.register_buffer(name, value, persistent)
#         else:
#             self.register_buffer(name, value)

#     def r(self):
#         return c2c(self.forward().v)

#     def forward(self, root_orient=None, pose_body=None, pose_hand=None, pose_jaw=None, pose_eye=None, betas=None,
#                 trans=None, dmpls=None, expression=None, v_template =None, joints=None, v_shaped=None, return_dict=False,  **kwargs):
#         '''

#         :param root_orient: Nx3
#         :param pose_body:
#         :param pose_hand:
#         :param pose_jaw:
#         :param pose_eye:
#         :param kwargs:
#         :return:
#         '''
#         batch_size = 1
#         # compute batchsize by any of the provided variables
#         for arg in [root_orient,pose_body,pose_hand,pose_jaw,pose_eye,betas,trans, dmpls,expression, v_template,joints]:
#             if arg is not None:
#                 batch_size = arg.shape[0]
#                 break

#         # assert not (v_template is not None and betas is not None), ValueError('vtemplate and betas could not be used jointly.')
#         assert self.model_type in ['smpl', 'smplh', 'smplx', 'mano', 'animal_horse', 'animal_dog', 'flame', 'animal_rat'], ValueError(
#             'model_type should be in smpl/smplh/smplx/mano')
#         if root_orient is None:  root_orient = self.init_root_orient.expand(batch_size, -1)
#         if self.model_type in ['smplh', 'smpl']:
#             if pose_body is None:  pose_body = self.init_pose_body.expand(batch_size, -1)
#             if pose_hand is None:  pose_hand = self.init_pose_hand.expand(batch_size, -1)
#         elif self.model_type == 'smplx':
#             if pose_body is None:  pose_body = self.init_pose_body.expand(batch_size, -1)
#             if pose_hand is None:  pose_hand = self.init_pose_hand.expand(batch_size, -1)
#             if pose_jaw is None:  pose_jaw = self.init_pose_jaw.expand(batch_size, -1)
#             if pose_eye is None:  pose_eye = self.init_pose_eye.expand(batch_size, -1)
#         elif self.model_type == 'flame':
#             if pose_body is None:  pose_body = self.init_pose_body.expand(batch_size, -1)
#             if pose_jaw is None:  pose_jaw = self.init_pose_jaw.expand(batch_size, -1)
#             if pose_eye is None:  pose_eye = self.init_pose_eye.expand(batch_size, -1)
#         elif self.model_type in ['mano',]:
#             if pose_hand is None:  pose_hand = self.init_pose_hand.expand(batch_size, -1)
#         elif self.model_type in ['animal_horse','animal_dog', 'animal_rat']:
#             if pose_body is None:  pose_body = self.init_pose_body.expand(batch_size, -1)

#         if pose_hand is None and self.model_type not in ['animal_horse', 'animal_dog', 'animal_rat','flame']:
#             pose_hand = self.init_pose_hand.expand(batch_size, -1)

#         if trans is None: trans = self.init_trans.expand(batch_size, -1)
#         if v_template is None: v_template = self.init_v_template.expand(batch_size, -1,-1)
#         if betas is None: betas = self.init_betas.expand(batch_size, -1)

#         if self.model_type in ['smplh', 'smpl']:
#             full_pose = torch.cat([root_orient, pose_body, pose_hand], dim=-1)
#         elif self.model_type == 'smplx':
#             full_pose = torch.cat([root_orient, pose_body, pose_jaw, pose_eye, pose_hand], dim=-1)  # orient:3, body:63, jaw:3, eyel:3, eyer:3, handl, handr
#         elif self.model_type == 'flame':
#             full_pose = torch.cat([root_orient, pose_body, pose_jaw, pose_eye], dim=-1)  # orient:3, body:63, jaw:3, eyel:3, eyer:3, handl, handr
#         elif self.model_type in ['mano', ]:
#             full_pose = torch.cat([root_orient, pose_hand], dim=-1)
#         elif self.model_type in ['animal_horse', 'animal_dog', 'animal_rat']:
#             full_pose = torch.cat([root_orient, pose_body], dim=-1)

#         if self.use_dmpl:
#             if dmpls is None: dmpls = self.init_dmpls.expand(batch_size, -1)
#             shape_components = torch.cat([betas, dmpls], dim=-1)
#             shapedirs = torch.cat([self.shapedirs, self.dmpldirs], dim=-1)
#         elif self.use_expression:
#             if expression is None: expression = self.init_expression.expand(batch_size, -1)
#             shape_components = torch.cat([betas, expression], dim=-1)
#             shapedirs = torch.cat([self.shapedirs, self.exprdirs], dim=-1)
#         else:
#             shape_components = betas
#             shapedirs = self.shapedirs

#         verts, Jtr = lbs(betas=shape_components, pose=full_pose, v_template=v_template,
#                             shapedirs=shapedirs, posedirs=self.posedirs,
#                             J_regressor=self.J_regressor, parents=self.kintree_table[0].long(),
#                             lbs_weights=self.weights, joints=joints, v_shaped=v_shaped,
#                             dtype=self.dtype)

#         Jtr = Jtr + trans.unsqueeze(dim=1)
#         verts = verts + trans.unsqueeze(dim=1)

#         res = {}
#         res['v'] = verts
#         res['f'] = self.f
#         res['Jtr'] = Jtr  # Todo: ik can be made with vposer
#         res['full_pose'] = full_pose

#         if not return_dict:
#             class result_meta(object):
#                 pass

#             res_class = result_meta()
#             for k, v in res.items():
#                 res_class.__setattr__(k, v)
#             res = res_class

#         return res