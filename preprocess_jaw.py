"""
this code extract jaw basis from all the training frames.
"""
import numpy as np 
import torch
import json
import os
import cv2 as cv

import sys 
sys.path.append('.')
from flame_model.flame import FlameHead
from mesh import TriMesh

def farthest_point_sampling(points):
    """
    :param points: torch.Tensor, (N, D), D is the dimension of each point
    :return: (N, )
    """
    rest_indices = list(range(points.shape[0]))
    sampled_indices = [0]
    rest_indices.remove(sampled_indices[0])
    while len(rest_indices) > 0:
        rest_points = points[rest_indices]
        sampled_points = points[sampled_indices]
        dot_dist = torch.abs(torch.einsum('vi,mi->vm', rest_points, sampled_points))  # larger is closer
        neg_dot_dist = 1. - dot_dist  # smaller is closer
        min_dist = neg_dot_dist.min(1)[0]
        argmax_pos = min_dist.argmax().item()
        max_idx = rest_indices[argmax_pos]
        sampled_indices.append(max_idx)
        del rest_indices[argmax_pos]
    return sampled_indices

def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    #copy from pytorch3d.transforms
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles

if __name__ == '__main__':
    flame_model = FlameHead(shape_params=300, expr_params=100)
    ids = ['074']
    data_types = ['val', 'test']
    intervals = [1, 16]
    flame_root = '' #please modify to your own dataset path, for example '/data1/wangyating/dataset/cluster/ikarus/sqian/project/dynamic-head-avatars/code/multi-view-head-tracker/export/'
    imgs = []
    for id in ids:
        id = id.strip('\n')
        print(id)
        transform_root = flame_root + 'UNION10_' + id + '/' #please modify to your own path
        
        jaw_poses = torch.FloatTensor([0,0,0,1.0])
        for i in range(len(data_types)):
            t, interval = data_types[i], intervals[i]
            data = json.load(open(transform_root + 'transforms_' + t + '.json', 'r'))
            for j in range(len(data['frames'])):
                if not j % interval == 0:
                    continue
                flame_npz = np.load(flame_root + os.path.join(*data['frames'][j]['flame_param_path'].split('/')[1:]))
                jaw_pose = torch.FloatTensor(flame_npz['jaw_pose'])
                jaw_pose = axis_angle_to_quaternion(jaw_pose)
                jaw_poses = torch.cat([jaw_poses, jaw_pose.flatten()])
        jaw_poses = jaw_poses.reshape((-1, 4))
        indices = farthest_point_sampling(jaw_poses)
        sorted_jaw_poses = jaw_poses[indices]
        np.savez(transform_root + 'sorted_jaw_poses.npz', jaw_poses = sorted_jaw_poses)