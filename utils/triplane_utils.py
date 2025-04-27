import torch
import itertools
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

relu = torch.nn.ReLU()

def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, feature_dim , align_corners: bool = True) -> torch.Tensor:
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)  

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))  
    
    interp_grid = grid[:, :feature_dim, :, :]
    B, feature_dim = interp_grid.shape[:2]
    n = coords.shape[-2]
    interp = grid_sampler(
        interp_grid,  
        coords, 
        align_corners=align_corners,
        mode='bilinear', padding_mode='border')  

    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  
    interp = interp.squeeze()  
    return interp

def init_planes(grid_dim, in_dim, out_dim, resolution, a, b):
    planes = list(itertools.combinations(range(in_dim), grid_dim))#plane:01 02 12
    plane_coefs = nn.ParameterList()
    for i, plane in enumerate(planes):
        if(i == 0):
            init_plane_coef = nn.Parameter(torch.empty([1, int(out_dim / 2)] + [resolution[cc] for cc in plane[::-1]])) 
        else:
            init_plane_coef = nn.Parameter(torch.empty([1, int(out_dim / 4)] + [resolution[cc] for cc in plane[::-1]]))  
        nn.init.uniform_(init_plane_coef, a=a, b=b)
        plane_coefs.append(init_plane_coef)
    return plane_coefs

def interpolate_ms_features(points, triplane, plane_dim, concat_f, num_levels):
    planes = list(itertools.combinations(range(points.shape[-1]), plane_dim)) 
    multi_scale_interp = [] if concat_f else 0.
    plane : nn.ParameterList
    for scale, plane in enumerate(triplane[:num_levels]):  
        interp_space = []
        for ci , coo_comb in enumerate(planes): 
            # plane[ci].shape: 1 feature_dim 64 64
            feature_dim = (plane[ci].shape[1])
            interp_out_plane = (grid_sample_wrapper(plane[ci], points[..., coo_comb], feature_dim).view(-1, feature_dim)) 
            # product reduce -> concat
            interp_space.append(interp_out_plane)
            # interp_space = interp_space * interp_out_plane

        interp_space = torch.cat(interp_space, dim=-1)
        if concat_f:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp = multi_scale_interp + interp_space

    if concat_f:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    return multi_scale_interp

class MLP(nn.Module):
    def __init__(self, mlptype, dim_in, dim_out, dim_hidden, num_layers):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.mlptype = mlptype

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=False))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = relu(x)
        return x
        
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim

class TriPlaneNetwork(nn.Module):
    def __init__(self, spatial_bounds, grid_dim=2, in_dim=3, out_dim=64, resolution=[128, 128, 128], a=-0.1, b=0.1):
        super().__init__()
        self.grid_dim = grid_dim 
        self.in_dim = in_dim  
        self.out_dim = out_dim  
        self.base_resolution = resolution  
        self.multi_scale_res = [1]
        self.concat_feature = False
        self.spatial_bounds = spatial_bounds

        assert self.in_dim == len(self.base_resolution), "Resolution must have same number of elements as input-dimension"
        self.tri_plane = nn.ModuleList()
        for i in range(len(self.multi_scale_res)):
            res_scale = self.multi_scale_res[i]
            resolution = [res_scale * resolution for resolution in self.base_resolution]
            plane_coefs = init_planes(self.grid_dim, self.in_dim, self.out_dim, resolution, a=a, b=b)

            if self.concat_feature:
                self.feature_dim += plane_coefs[-1].shape[1] 
            else:
                self.feature_dim = 32 + 16 + 16
            self.tri_plane.append(plane_coefs)

        self.pe, self.view_dim = get_embedder(4)
        self.hidden_dim_view = 128
        self.num_layers_view = 2 
        self.color_out_dim = 3 
        self.shs_net = MLP(mlptype='color', dim_in=self.view_dim + self.feature_dim, dim_out=self.color_out_dim,
                           dim_hidden=self.hidden_dim_view, num_layers=self.num_layers_view)
       
        self.interpolate_times = []
        self.mlp_times = []
        
    def forward(self, xyz, dirs):  
        assert len(xyz.shape) == 2 and xyz.shape[-1] == self.in_dim, 'input points dim must be (num_points, 3)'

        xyz_norm = (xyz - self.spatial_bounds[0]) / (self.spatial_bounds[1] - self.spatial_bounds[0]) * 2 - 1
        canonical_f = interpolate_ms_features(xyz_norm, self.tri_plane, plane_dim=self.grid_dim, concat_f=self.concat_feature, num_levels=None)  

        dirs_pe = self.pe(dirs)
        input = torch.cat([canonical_f, dirs_pe], dim=-1)
        color = self.shs_net(input)  # only geo_feat, no dirs will causing a sharp drop in performance
        return color.reshape((-1, 1, 3))

    def get_tvloss(self):
        tv_loss = 0.0
        for triplane in self.tri_plane:
            for plane in triplane:
                tv_loss += torch.sum(torch.abs(plane[:, :, :, :-1] - plane[:, :, :, 1:]))
                + torch.sum(torch.abs(plane[:, :, :-1, :] - plane[:, :, 1:, :]))
        return tv_loss / (len(self.tri_plane) * len(triplane))
