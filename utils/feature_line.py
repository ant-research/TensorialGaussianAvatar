import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import logging
from datetime import datetime

def sample_feature_line(bs_feat_line, jaw_feat_line, pts):
    l, c = bs_feat_line.shape
    pts = torch.clamp(pts, 0, 1) * (l - 1)

    left_index = torch.floor(pts).long()
    right_index = torch.clamp(left_index + 1, max = l - 1)
    weight = pts - left_index.float()

    bs_l = bs_feat_line[left_index]
    bs_r = bs_feat_line[right_index]
    bs_interp = torch.lerp(bs_l, bs_r, weight[:, None])

    jaw_l = jaw_feat_line[left_index]
    jaw_r = jaw_feat_line[right_index]
    jaw_interp = torch.lerp(jaw_l, jaw_r, weight[:, None])

    return bs_interp, jaw_interp

class FeatureLine(nn.Module):
    def __init__(self, expr_num = 80, key_jaw_pose_num = 16, line_size = (64, 64, 64), feat_dim = [32, 32, 32], n_layer = 2, n_hidden = 128):
        super(FeatureLine, self).__init__()

        self.expr_num = expr_num
        self.Lx, self.Ly, self.Lz = line_size
        self.Cx, self.Cy, self.Cz = feat_dim 
        self.key_jaw_pose_num = key_jaw_pose_num

        feat_lines_x = torch.zeros((self.expr_num + self.key_jaw_pose_num, self.Lx, self.Cx), dtype = torch.float32)
        self.register_parameter('feat_lines_x', nn.Parameter(feat_lines_x))
        nn.init.constant_(self.feat_lines_x.data, 0.)

        feat_lines_y = torch.zeros((self.expr_num + self.key_jaw_pose_num, self.Ly, self.Cy), dtype = torch.float32)
        self.register_parameter('feat_lines_y', nn.Parameter(feat_lines_y))
        nn.init.constant_(self.feat_lines_y.data, 0.)

        feat_lines_z = torch.zeros((self.expr_num + self.key_jaw_pose_num, self.Lz, self.Cz), dtype = torch.float32)
        self.register_parameter('feat_lines_z', nn.Parameter(feat_lines_z))
        nn.init.constant_(self.feat_lines_z.data, 0.)

        self.n_layer = n_layer 
        self.input_dims = [(self.Cx + self.Cy + self.Cz) * 2] + [n_hidden for _ in range(n_layer)] + [1]
        for l in range(len(self.input_dims) - 1):
            in_dim, out_dim = self.input_dims[l], self.input_dims[l + 1]
            lin = nn.Linear(in_dim, out_dim)
            lin = nn.utils.weight_norm(lin)
            setattr(self, "layer_" + str(l), lin)

        self.activation = nn.ReLU()
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain = 1)
    
    def forward(self, expr, jaw_quat_weight, xyz):   
        expr = expr.flatten()[:self.expr_num].reshape((self.expr_num, 1))

        bs_fl_x = torch.einsum('ijk,ik->jk', self.feat_lines_x[:self.expr_num], expr)
        bs_fl_y = torch.einsum('ijk,ik->jk', self.feat_lines_y[:self.expr_num], expr)
        bs_fl_z = torch.einsum('ijk,ik->jk', self.feat_lines_z[:self.expr_num], expr)

        jaw_fl_x = torch.einsum('ijk,ik->jk', self.feat_lines_x[self.expr_num:], jaw_quat_weight)
        jaw_fl_y = torch.einsum('ijk,ik->jk', self.feat_lines_y[self.expr_num:], jaw_quat_weight)
        jaw_fl_z = torch.einsum('ijk,ik->jk', self.feat_lines_z[self.expr_num:], jaw_quat_weight)

        bs_x, jaw_x = sample_feature_line(bs_fl_x, jaw_fl_x, xyz[:, 0])
        bs_y, jaw_y = sample_feature_line(bs_fl_y, jaw_fl_y, xyz[:, 1])
        bs_z, jaw_z = sample_feature_line(bs_fl_z, jaw_fl_z, xyz[:, 2])

        fea = torch.cat((bs_x, bs_y, bs_z, jaw_x, jaw_y, jaw_z), dim = 1)
        for l in range(len(self.input_dims) - 1):
            lin = getattr(self, 'layer_' + str(l))
            fea = lin(fea)
            if l < len(self.input_dims) - 2:
                fea = self.activation(fea)
        
        return fea
