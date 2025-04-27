"""
this code cluster training expressions into 16 categories.
"""
import numpy as np 
import matplotlib.pyplot as plt
import cv2 as cv
import os
import json
from mesh import TriMesh
import sys 
sys.path.append('.')
from flame_model.flame import FlameHead
from flame_model.lbs import batch_rodrigues
import torch
import pickle
from sklearn.cluster import KMeans

import cv2 as cv

from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.cluster import SpectralClustering

class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

def custom_distance(rest_points, sampled_points, batch_size=64):
    M, K, D = rest_points.shape[0], sampled_points.shape[0], rest_points.shape[1]
    mean_distances = torch.zeros(M, K, device=rest_points.device)

    for i in range(0, M, batch_size):
        batch_rest_points = rest_points[i:i + batch_size]  # (batch_size, 5143, 3)

        diff = batch_rest_points.unsqueeze(1) - sampled_points.unsqueeze(0)  # (batch_size, K, 5143, 3)
        squared_diff = diff ** 2  # (batch_size, K, 5143, 3)
        distances = torch.sqrt(squared_diff.sum(dim=-1))  # (batch_size, K, 5143)

        mean_distances[i:i + batch_size] = distances.mean(dim=-1)  # (batch_size, K)

    return mean_distances

def farthest_point_sampling(points, num_samples=16):
    device = points.device
    rest_indices = list(range(points.shape[0]))
    sampled_indices = [0]  # Start by sampling the first point
    rest_indices.remove(sampled_indices[0])
    
    while len(rest_indices) > points.shape[0] - num_samples:
        rest_points = points[rest_indices]
        sampled_points = points[sampled_indices]

        distances = custom_distance(rest_points, sampled_points)  # 维度 (M, K)
        min_dist = distances.min(dim=1)[0]  # (M,)
        farthest_idx = min_dist.argmax().item()
        
        max_idx = rest_indices[farthest_idx]
        sampled_indices.append(max_idx)
        del rest_indices[farthest_idx]
    
    return sampled_indices

def clustering(points, num_clusters):
    distance_matrix = custom_distance(points, points)
    similarity_matrix = 1 - distance_matrix / distance_matrix.max()
    
    spectral_clustering = SpectralClustering(n_clusters=num_clusters, affinity='precomputed')
    labels = spectral_clustering.fit_predict(similarity_matrix.cpu().numpy())

    centers = []
    for i in range(num_clusters):
        centers.append(points[np.where(labels == i)[0]].mean(axis = 0).cpu().numpy())

    return centers, labels


ids = ['074']

flame_model = FlameHead(shape_params=300,expr_params=100)
with open('flame_model/assets/flame/FLAME_masks.pkl', "rb") as f:
    ss = pickle.load(f, encoding="latin1")
    flame_masks = Struct(**ss)
    eye_in_flame = torch.from_numpy(flame_masks.eye_region)
    face_in_flame = torch.from_numpy(flame_masks.face)

vmean = flame_model.v_template
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors = 1)
neigh.fit(vmean[face_in_flame].numpy())
eye = vmean[eye_in_flame].numpy()
eye_in_face = neigh.kneighbors(eye, return_distance = False).flatten()

template = TriMesh()
template.load('flame_model/assets/flame/head_template_mesh.obj')
flame_root = '' #please modify to your own dataset path, for example '/dataset/cluster/ikarus/sqian/project/dynamic-head-avatars/code/multi-view-head-tracker/export/'
for id in ids:
    print(id)
    verts_neutral = None
    transform_root = flame_root + 'UNION10_' + id + '/' #please modify this path to your own
    data = json.load(open(transform_root + 'transforms_train.json', 'r'))
    point_lst = []
    for j in range(len(data['frames'])):
        camera_index = data['frames'][j]['camera_index']
        if not camera_index == 0:
            continue
        flame_npz = np.load(flame_root + os.path.join(*data['frames'][j]['flame_param_path'].split('/')[1:]))

        flame_param = {key: value for key, value in flame_npz.items()}
        for key, value in flame_param.items():
            flame_param[key] = torch.FloatTensor(value)
        translation, rotation = flame_param['translation'], flame_param['rotation']
        neck_pose, jaw_pose, eyes_pose = flame_param['neck_pose'], flame_param['jaw_pose'], flame_param['eyes_pose']
        shape, expr = flame_param['shape'][None, :], flame_param['expr']
        static_offset = flame_param['static_offset']
        verts_unposed, _, J_unposed = flame_model(shape, expr, rotation * 0, neck_pose * 0, jaw_pose, eyes_pose, translation * 0, static_offset = static_offset)
        if verts_neutral is None:
            verts_neutral, _, J = flame_model(shape, expr * 0, rotation * 0, neck_pose * 0, jaw_pose*0, eyes_pose*0, translation * 0, static_offset = static_offset)

        bias = verts_unposed - verts_neutral
        bias[:, eye_in_flame, :] *= 2 #add weight to eye region vertices when calculating face motion
        bias = bias[:, face_in_flame, :]
        point_lst.append(bias)
        
    point_lst = torch.cat(point_lst, dim=0).cuda()
    # 
    centers, labels = clustering(point_lst, 16) #classes number
    np.savez(transform_root + 'train_cluster_16.npz', centers = centers, labels = labels)