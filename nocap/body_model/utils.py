import os
import numpy as np
import torch

SMPL_JOINTS = {'hips' : 0, 'leftUpLeg' : 1, 'rightUpLeg' : 2, 'spine' : 3, 'leftLeg' : 4, 'rightLeg' : 5,
                'spine1' : 6, 'leftFoot' : 7, 'rightFoot' : 8, 'spine2' : 9, 'leftToeBase' : 10, 'rightToeBase' : 11, 
                'neck' : 12, 'leftShoulder' : 13, 'rightShoulder' : 14, 'head' : 15, 'leftArm' : 16, 'rightArm' : 17,
                'leftForeArm' : 18, 'rightForeArm' : 19, 'leftHand' : 20, 'rightHand' : 21}
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12, 13, 14, 16, 17, 18, 19]

SMPLH_PATH = './body_models/smplh'
SMPLX_PATH = './body_models/smplx'
SMPL_PATH = './body_models/smpl'
VPOSER_PATH = './body_models/vposer_v1_0'

# chosen virtual mocap markers that are "keypoints" to work with
KEYPT_VERTS = [4404, 920, 3076, 3169, 823, 4310, 1010, 1085, 4495, 4569, 6615, 3217, 3313, 6713,
            6785, 3383, 6607, 3207, 1241, 1508, 4797, 4122, 1618, 1569, 5135, 5040, 5691, 5636,
            5404, 2230, 2173, 2108, 134, 3645, 6543, 3123, 3024, 4194, 1306, 182, 3694, 4294, 744]


# #
# # From https://github.com/vchoutas/smplify-x/blob/master/smplifyx/utils.py
# # Please see license for usage restrictions.
# #
# def smpl_to_openpose(model_type='smplx', use_hands=True, use_face=True,
#                      use_face_contour=False, openpose_format='coco25'):
#     ''' Returns the indices of the permutation that maps SMPL to OpenPose

#         Parameters
#         ----------
#         model_type: str, optional
#             The type of SMPL-like model that is used. The default mapping
#             returned is for the SMPLX model
#         use_hands: bool, optional
#             Flag for adding to the returned permutation the mapping for the
#             hand keypoints. Defaults to True
#         use_face: bool, optional
#             Flag for adding to the returned permutation the mapping for the
#             face keypoints. Defaults to True
#         use_face_contour: bool, optional
#             Flag for appending the facial contour keypoints. Defaults to False
#         openpose_format: bool, optional
#             The output format of OpenPose. For now only COCO-25 and COCO-19 is
#             supported. Defaults to 'coco25'

#     '''
#     if openpose_format.lower() == 'coco25':
#         if model_type == 'smpl':
#             return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
#                              7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
#                             dtype=np.int32)
#         elif model_type == 'smplh':
#             body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
#                                      8, 1, 4, 7, 53, 54, 55, 56, 57, 58, 59,
#                                      60, 61, 62], dtype=np.int32)
#             mapping = [body_mapping]
#             if use_hands:
#                 lhand_mapping = np.array([20, 34, 35, 36, 63, 22, 23, 24, 64,
#                                           25, 26, 27, 65, 31, 32, 33, 66, 28,
#                                           29, 30, 67], dtype=np.int32)
#                 rhand_mapping = np.array([21, 49, 50, 51, 68, 37, 38, 39, 69,
#                                           40, 41, 42, 70, 46, 47, 48, 71, 43,
#                                           44, 45, 72], dtype=np.int32)
#                 mapping += [lhand_mapping, rhand_mapping]
#             return np.concatenate(mapping)
#         # SMPLX
#         elif model_type == 'smplx':
#             body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
#                                      8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
#                                      63, 64, 65], dtype=np.int32)
#             mapping = [body_mapping]
#             if use_hands:
#                 lhand_mapping = np.array([20, 37, 38, 39, 66, 25, 26, 27,
#                                           67, 28, 29, 30, 68, 34, 35, 36, 69,
#                                           31, 32, 33, 70], dtype=np.int32)
#                 rhand_mapping = np.array([21, 52, 53, 54, 71, 40, 41, 42, 72,
#                                           43, 44, 45, 73, 49, 50, 51, 74, 46,
#                                           47, 48, 75], dtype=np.int32)

#                 mapping += [lhand_mapping, rhand_mapping]
#             if use_face:
#                 #  end_idx = 127 + 17 * use_face_contour
#                 face_mapping = np.arange(76, 127 + 17 * use_face_contour,
#                                          dtype=np.int32)
#                 mapping += [face_mapping]

#             return np.concatenate(mapping)
#         else:
#             raise ValueError('Unknown model type: {}'.format(model_type))
#     elif openpose_format == 'coco19':
#         if model_type == 'smpl':
#             return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8,
#                              1, 4, 7, 25, 26, 27, 28],
#                             dtype=np.int32)
#         elif model_type == 'smplh':
#             body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
#                                      8, 1, 4, 7, 53, 54, 55, 56],
#                                     dtype=np.int32)
#             mapping = [body_mapping]
#             if use_hands:
#                 lhand_mapping = np.array([20, 34, 35, 36, 57, 22, 23, 24, 58,
#                                           25, 26, 27, 59, 31, 32, 33, 60, 28,
#                                           29, 30, 61], dtype=np.int32)
#                 rhand_mapping = np.array([21, 49, 50, 51, 62, 37, 38, 39, 63,
#                                           40, 41, 42, 64, 46, 47, 48, 65, 43,
#                                           44, 45, 66], dtype=np.int32)
#                 mapping += [lhand_mapping, rhand_mapping]
#             return np.concatenate(mapping)
#         # SMPLX
#         elif model_type == 'smplx':
#             body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
#                                      8, 1, 4, 7, 56, 57, 58, 59],
#                                     dtype=np.int32)
#             mapping = [body_mapping]
#             if use_hands:
#                 lhand_mapping = np.array([20, 37, 38, 39, 60, 25, 26, 27,
#                                           61, 28, 29, 30, 62, 34, 35, 36, 63,
#                                           31, 32, 33, 64], dtype=np.int32)
#                 rhand_mapping = np.array([21, 52, 53, 54, 65, 40, 41, 42, 66,
#                                           43, 44, 45, 67, 49, 50, 51, 68, 46,
#                                           47, 48, 69], dtype=np.int32)

#                 mapping += [lhand_mapping, rhand_mapping]
#             if use_face:
#                 face_mapping = np.arange(70, 70 + 51 +
#                                          17 * use_face_contour,
#                                          dtype=np.int32)
#                 mapping += [face_mapping]

#             return np.concatenate(mapping)
#         else:
#             raise ValueError('Unknown model type: {}'.format(model_type))
#     else:
#         raise ValueError('Unknown joint format: {}'.format(openpose_format))
    

def vertices2joints(J_regressor, vertices):
    ''' Calculates the 3D joint locations from the vertices

    Parameters
    ----------
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : torch.tensor BxVx3
        The tensor of mesh vertices

    Returns
    -------
    torch.tensor BxJx3
        The location of the joints
    '''

    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])


def blend_shapes(betas, shape_disps):
    ''' Calculates the per vertex displacement due to the blend shapes


    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    '''

    # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
    # i.e. Multiply each shape displacement by its corresponding beta and
    # then sum them.

    #print(betas.device,shape_disps.device)
    blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
    return blend_shape



def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms
    

def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, parents,
        lbs_weights, joints = None, pose2rot=True, v_shaped=None, dtype=torch.float32):
    ''' Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    '''

    batch_size = max(betas.shape[0], pose.shape[0])
    device = betas.device

    # Add shape contribution
    if v_shaped is None:
        v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    if joints is not None:
        J = joints
    else:
        J = vertices2joints(J_regressor, v_shaped)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(
            pose.view(-1, 3), dtype=dtype).view([batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(pose_feature, posedirs).view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    posedirs).view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped
    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed