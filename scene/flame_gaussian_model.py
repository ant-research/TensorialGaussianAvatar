# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from flame_model.flame import FlameHead

from .gaussian_model import GaussianModel
from utils.feature_line import FeatureLine
from utils.triplane_utils import TriPlaneNetwork

from utils.graphics_utils import compute_face_orientation
# from pytorch3d.transforms import matrix_to_quaternion
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz

import pickle
class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6

    small_angles = angles.abs() < eps
    
    sin_half_angles_over_angles = torch.where(
        small_angles,
        0.5 - (angles * angles) / 48,
        torch.sin(half_angles) / (angles + eps) 
    )

    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    
    return quaternions

class FlameGaussianModel(GaussianModel):
    def __init__(self, source_path, sh_degree : int, disable_flame_static_offset=False, not_finetune_flame_params=False, \
        static = 'triplane', dynamic = 'FeatureLine', \
        n_shape=300, n_expr=100):
        super().__init__(sh_degree, static = static, dynamic = dynamic)

        self.disable_flame_static_offset = disable_flame_static_offset
        self.not_finetune_flame_params = not_finetune_flame_params

        self.static = static
        self.dynamic = dynamic

        self.n_shape = n_shape
        self.n_expr = n_expr
        self.source_path = source_path

        self.flame_model = FlameHead(
            n_shape, 
            n_expr,
            add_teeth=True,
        ).cuda()
        self.flame_param = None
        self.flame_param_orig = None

        # binding is initialized once the mesh topology is known
        if self.binding is None:
            self.binding = torch.arange(len(self.flame_model.faces)).cuda()
            self.binding_counter = torch.ones(len(self.flame_model.faces), dtype=torch.int32).cuda()
        
        key_jaw_poses_path = self.source_path + '/sorted_jaw_poses.npz'
        self.key_jaw_pose_num = 16
        self.key_jaw_poses = torch.from_numpy(np.load(key_jaw_poses_path)['jaw_poses'][:self.key_jaw_pose_num]).cuda()
        
        with open('./flame_model/assets/flame/FLAME_masks.pkl', "rb") as f:
            ss = pickle.load(f, encoding="latin1")
            self.flame_masks = Struct(**ss)

            self.vert_in_face = torch.from_numpy(self.flame_masks.face).cuda()
            self.tri_in_face = torch.where(torch.all(torch.isin(self.flame_model.faces, self.vert_in_face), dim = -1))[0]
    
    def update_cano_mesh(self):
        flame_param = self.flame_param

        verts, verts_cano = self.flame_model(
            flame_param['shape'][None, ...],
            torch.zeros_like(flame_param['expr'][[0]]).cuda(),
            torch.zeros_like(flame_param['rotation'][[0]]).cuda(),
            torch.zeros_like(flame_param['neck_pose'][[0]]).cuda(),
            torch.zeros_like(flame_param['jaw_pose'][[0]]).cuda(),
            torch.zeros_like(flame_param['eyes_pose'][[0]]).cuda(),
            torch.zeros_like(flame_param['translation'][[0]]).cuda(),
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=flame_param['static_offset'],
            dynamic_offset=None,
        )

        self.verts_cano = verts
        self.cano_face_max = torch.max(self.verts_cano[0].detach()[self.vert_in_face], dim = 0)[0]
        self.cano_face_min = torch.min(self.verts_cano[0].detach()[self.vert_in_face], dim = 0)[0]
        self.cano_head_max = torch.max(self.verts_cano[0].detach(), dim = 0)[0]
        self.cano_head_min = torch.min(self.verts_cano[0].detach(), dim = 0)[0]

        self.face_center_cano = verts[:, self.flame_model.faces].mean(dim=-2).squeeze(0)
        self.face_orien_mat_cano, self.face_scaling_cano = compute_face_orientation(verts_cano.squeeze(0), self.flame_model.faces.squeeze(0), return_scale=True)

    def load_meshes(self, train_meshes, test_meshes, tgt_train_meshes, tgt_test_meshes):
        if self.flame_param is None:
            meshes = {**train_meshes, **test_meshes}
            tgt_meshes = {**tgt_train_meshes, **tgt_test_meshes}
            pose_meshes = meshes if len(tgt_meshes) == 0 else tgt_meshes
            
            self.num_timesteps = max(pose_meshes) + 1  # required by viewers
            num_verts = self.flame_model.v_template.shape[0]

            if not self.disable_flame_static_offset:
                static_offset = torch.from_numpy(meshes[0]['static_offset'])
                if static_offset.shape[0] != num_verts:
                    static_offset = torch.nn.functional.pad(static_offset, (0, 0, 0, num_verts - meshes[0]['static_offset'].shape[1]))
            else:
                static_offset = torch.zeros([num_verts, 3])

            T = self.num_timesteps

            self.flame_param = {
                'shape': torch.from_numpy(meshes[0]['shape']),
                'expr': torch.zeros([T, (meshes[0]['expr']).shape[1]]),
                'rotation': torch.zeros([T, 3]),
                'neck_pose': torch.zeros([T, 3]),
                'jaw_pose': torch.zeros([T, 3]),
                'jaw_quat_weight': torch.zeros([T, 16]),
                'eyes_pose': torch.zeros([T, 6]),
                'translation': torch.zeros([T, 3]),
                'static_offset': static_offset
            }

            for i, mesh in pose_meshes.items():
                self.flame_param['expr'][i] = torch.from_numpy(mesh['expr'])
                self.flame_param['rotation'][i] = torch.from_numpy(mesh['rotation'])
                self.flame_param['neck_pose'][i] = torch.from_numpy(mesh['neck_pose'])
                self.flame_param['jaw_pose'][i] = torch.from_numpy(mesh['jaw_pose'])
                self.flame_param['eyes_pose'][i] = torch.from_numpy(mesh['eyes_pose'])
                self.flame_param['translation'][i] = torch.from_numpy(mesh['translation'])
                # self.flame_param['dynamic_offset'][i] = torch.from_numpy(mesh['dynamic_offset'])
            
            for k, v in self.flame_param.items():
                self.flame_param[k] = v.float().cuda()
            
            self.flame_param_orig = {k: v.clone() for k, v in self.flame_param.items()}

            self.update_cano_mesh()

        else:
            # NOTE: not sure when this happens
            import ipdb; ipdb.set_trace()
            pass
            
        if self.dynamic == 'FeatureLine':
            # Feature lines are designed to model dynamic textures caused by expressions, actually hair and neck should not be included in feature lines
            # but in experiments we found including hair and neck led to fewer splats thus accelerating rendering
            # We try to split feature lines for skin area and non-skin area, which enhancing robustness
            # but this design leads to additional time and space cost, and gets no improvement on PSNR metrics

            # self.feat_line_face = FeatureLine(expr_num = 50).cuda()
            # self.feat_line_head = FeatureLine(expr_num = 30).cuda()
            self.feat_line = FeatureLine(expr_num = 80).cuda()

        if self.static == 'triplane':
            # we manually set the spatial range of triplanes by observing flame template mesh
            # for custom face shape prior model, this range should be adjusted
            bound_upper = torch.max(self.verts_cano[0].detach(), dim = 0)[0] + 0.005
            bound_lower = torch.min(self.verts_cano[0].detach(), dim = 0)[0] - 0.005
            self.trip_spatial_bounds = torch.cat([bound_lower, bound_upper], dim = 0).reshape((2, 3))
            self.trip_spatial_bounds[0, 1] -= 0.025
            self.trip_spatial_bounds[0, 0] = -0.25
            self.trip_spatial_bounds[1, 0] = 0.25
            self.triplane = TriPlaneNetwork(self.trip_spatial_bounds).cuda()
    
    def prepare_meshes(self):
        # precompute mesh geometry and jaw pose weights in the initial stage to reduce time cost while inference
        T = self.num_timesteps
        
        self.mesh_lst = torch.zeros([T, 5143, 3], requires_grad=False).cuda() 

        self.face_center_lst = torch.zeros([T, 10144, 3], requires_grad=False).cuda()
        self.face_orien_mat_lst = torch.zeros([T, 10144, 3, 3], requires_grad=False).cuda()
        self.face_orien_quat_lst = torch.zeros([T, 10144, 4], requires_grad=False).cuda()
        self.face_scaling_lst = torch.zeros([T, 10144, 1], requires_grad=False).cuda()

        self.jaw_quat_weight_lst = torch.zeros([T, self.key_jaw_poses.shape[0], 1], requires_grad = False).cuda()

        faces = self.flame_model.faces

        with torch.no_grad():
            for i in range(self.num_timesteps):
                verts, verts_cano = self.flame_model(
                    self.flame_param['shape'][None, ...],
                    self.flame_param['expr'][[i]],
                    self.flame_param['rotation'][[i]],
                    self.flame_param['neck_pose'][[i]],
                    self.flame_param['jaw_pose'][[i]],
                    self.flame_param['eyes_pose'][[i]],
                    self.flame_param['translation'][[i]],
                    zero_centered_at_root_node=False,
                    return_landmarks=False,
                    return_verts_cano=True,
                    static_offset=self.flame_param['static_offset'],
                    dynamic_offset=None
                )

                self.mesh_lst[i] = verts  
                triangles = verts[:, faces]
                self.face_center_lst[i] = triangles.mean(dim=-2).squeeze(0)
                self.face_orien_mat_lst[i], self.face_scaling_lst[i] = compute_face_orientation(verts.squeeze(0), faces.squeeze(0), return_scale=True)
                self.face_orien_quat_lst[i] = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat_lst[i]))  # roma

                jaw_pose = self.flame_param['jaw_pose'][[i]].detach()
                jaw_pose_quat = axis_angle_to_quaternion(jaw_pose)
                quat_sim = torch.abs((jaw_pose_quat * self.key_jaw_poses).sum(-1))
                jaw_quat_weight = F.normalize(quat_sim, dim = 0, p = 1, eps = 1e-16)[:, None]
                self.jaw_quat_weight_lst[i] = jaw_quat_weight
    
    def select_mesh_by_timestep(self, timestep, original=False):
        self.timestep = timestep
        self.xyz_cano = self.get_xyz_cano

        if self.not_finetune_flame_params:
            verts = self.mesh_lst[timestep][None]
            self.face_center = self.face_center_lst[timestep]
            self.face_orien_mat = self.face_orien_mat_lst[timestep]
            self.face_orien_quat = self.face_orien_quat_lst[timestep]
            self.face_scaling = self.face_scaling_lst[timestep]
        else:
            flame_param = self.flame_param
            verts, verts_cano = self.flame_model(
                flame_param['shape'][None, ...],
                flame_param['expr'][[timestep]],
                flame_param['rotation'][[timestep]],
                flame_param['neck_pose'][[timestep]],
                flame_param['jaw_pose'][[timestep]],
                flame_param['eyes_pose'][[timestep]],
                flame_param['translation'][[timestep]],
                zero_centered_at_root_node=False,
                return_landmarks=False,
                return_verts_cano=True,
                static_offset=flame_param['static_offset'],
                dynamic_offset=None,
            )
            faces = self.flame_model.faces
            triangles = verts[:, faces]

            self.face_center = triangles.mean(dim=-2).squeeze(0)
            self.face_orien_mat, self.face_scaling = compute_face_orientation(verts.squeeze(0), faces.squeeze(0), return_scale=True)
            self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat))  # roma
    
    def loss_flame_param_normalize(self, timestep):
        # we utilize flame pca expression parameters to interpolate feature lines, gaussianavatars optimize pca coefficients simultaneously
        # using the optimized pca coefficients to interpolate feature lines causes a very unstable self-reenactment results.
        # I guess using semantic blendshapes(e.g. FACS) coefficients rather than pca coefficients is more suitable for feature line interpolation
        # but we have no open-source blendshapes and corresponding tracker
        # so we constrain pca parameters to be the same or disable flame pca coef optimization
        loss_expr = torch.abs(self.flame_param['expr'][timestep] - self.flame_param_orig['expr'][timestep]).mean()
        
        jaw_pose_quat = axis_angle_to_quaternion(self.flame_param['jaw_pose'][timestep])
        jaw_pose_quat_ori = axis_angle_to_quaternion(self.flame_param_orig['jaw_pose'][timestep])
        loss_jaw_pose = 1 - torch.abs((jaw_pose_quat * jaw_pose_quat_ori).sum(-1))

        eye_pose_quat = axis_angle_to_quaternion(self.flame_param['eyes_pose'][timestep])
        eye_pose_quat_ori = axis_angle_to_quaternion(self.flame_param_orig['eyes_pose'][timestep])
        loss_eye_pose = 1 - torch.abs((eye_pose_quat * eye_pose_quat_ori).sum(-1))

        return loss_expr + loss_jaw_pose + loss_eye_pose

    def loss_offset_normalize(self, timestep):
        with torch.no_grad():
            flame_param = self.flame_param
            verts, verts_cano = self.flame_model(
                    self.flame_param['shape'][None, ...],
                    self.flame_param['expr'][[timestep]],
                    self.flame_param['rotation'][[timestep]] * 0,
                    self.flame_param['neck_pose'][[timestep]] * 0,
                    self.flame_param['jaw_pose'][[timestep]],
                    self.flame_param['eyes_pose'][[timestep]],
                    self.flame_param['translation'][[timestep]] * 0,
                    zero_centered_at_root_node=False,
                    return_landmarks=False,
                    return_verts_cano=True,
                    static_offset=self.flame_param['static_offset'],
                    dynamic_offset=None
                )
            faces = self.flame_model.faces

            tri_unposed, tri = verts[0][faces], self.verts_cano[0][faces]
            center_unposed, center = tri_unposed.mean(dim=-2), tri.mean(dim=-2)
            tri_dist = torch.sqrt(torch.sum((center_unposed - center) ** 2, dim = -1))
            gs_dist = tri_dist[self.binding]

            mask_face = torch.isin(self.binding, self.tri_in_face)

            # we set different threshold and weight for different region
            weight = torch.zeros_like(gs_dist)
            weight[torch.logical_and(mask_face, gs_dist < 1e-3)] = 1
            weight[torch.logical_and(~mask_face, gs_dist < 2e-3)] = 10

        loss = (torch.abs(self._opacity_offset.flatten()) * weight).mean()
        return loss
    
    def update_offset(self):
        timestep = self.timestep
        xyz_cano = self.xyz_cano.detach()

        if self.not_finetune_flame_params:
            jaw_quat_weight = self.jaw_quat_weight_lst[timestep]
        else:
            jaw_pose = self.flame_param['jaw_pose'][[timestep]].detach()
            jaw_pose_quat = axis_angle_to_quaternion(jaw_pose)
            quat_sim = torch.abs((jaw_pose_quat * self.key_jaw_poses).sum(-1))
            jaw_quat_weight = F.normalize(quat_sim, dim = 0, p = 1, eps = 1e-16)[:, None]
        
        mask = ((xyz_cano >= self.cano_head_min) & (xyz_cano <= self.cano_head_max)).all(dim = 1)
        xyz_norm = (xyz_cano[mask] - self.cano_head_min) / (self.cano_head_max - self.cano_head_min)

        self._opacity_offset = torch.zeros_like(xyz_cano[:, 0])[:, None]
        self._opacity_offset[mask] = self.feat_line(self.flame_param['expr'][[timestep]].detach(), jaw_quat_weight, xyz_norm.detach())

        # mask_face = ((xyz_cano >= self.cano_face_min) & (xyz_cano <= self.cano_face_max)).all(dim = 1)
        # mask_head = ((xyz_cano >= self.cano_head_min) & (xyz_cano <= self.cano_head_max)).all(dim = 1)
        # mask_head[mask_face] = False

        # self._opacity_offset = torch.zeros_like(xyz_cano[:, 0])[:, None]
        # xyz_norm_face = (xyz_cano[mask_face] - self.cano_face_min) / (self.cano_face_max - self.cano_face_min)
        # self._opacity_offset[mask_face] = self.feat_line_face(self.flame_param['expr'][[timestep]].detach(), jaw_quat_weight, xyz_norm_face.detach())
        # xyz_norm_head = (xyz_cano[mask_head] - self.cano_head_min) / (self.cano_head_max - self.cano_head_min)
        # self._opacity_offset[mask_head] = self.feat_line_head(self.flame_param['expr'][[timestep]].detach(), jaw_quat_weight, xyz_norm_head.detach())

    def training_setup(self, training_args):
        super().training_setup(training_args)

        if self.dynamic == 'FeatureLine':
            param_fl_decoder, param_fl_xyz = [], []
            for name, param in self.feat_line.named_parameters():
                if 'layer_' in name:
                    param_fl_decoder.append(param)
                elif 'feat_lines_' in name:
                    param_fl_xyz.append(param)
            # for name, param in self.feat_line_face.named_parameters():
            #     if 'layer_' in name:
            #         param_fl_decoder.append(param)
            #     elif 'feat_lines_' in name:
            #         param_fl_xyz.append(param)
            
            # for name, param in self.feat_line_head.named_parameters():
            #     if 'layer_' in name:
            #         param_fl_decoder.append(param)
            #     elif 'feat_lines_' in name:
            #         param_fl_xyz.append(param)

            self.optimizer.add_param_group({'params': param_fl_decoder, 'lr': 1e-4, 'name': 'fl_decoder'})
            self.optimizer.add_param_group({'params': param_fl_xyz, 'lr': 2e-3, 'name': 'fl_xyz'})

        if self.static == 'triplane':
            param_triplane_xyz = []
            for module in self.triplane.tri_plane:
                param_triplane_xyz.extend(module.parameters())
            self.optimizer.add_param_group({"params": param_triplane_xyz, 'lr': 2e-3, 'name':"trip"})
            self.optimizer.add_param_group({"params": list(self.triplane.shs_net.parameters()), 'lr': 1e-4, 'name': 'trip.shs'})


        if self.not_finetune_flame_params:
            return
        
        ###########################################################################
        # self.flame_param['rotation'].requires_grad = True
        # self.flame_param['neck_pose'].requires_grad = True
        # self.flame_param['jaw_pose'].requires_grad = True
        # self.flame_param['eyes_pose'].requires_grad = True
        # params = [
        #     self.flame_param['rotation'],
        #     self.flame_param['neck_pose'],
        #     self.flame_param['jaw_pose'],
        #     self.flame_param['eyes_pose'],
        # ]
        # param_pose = {'params': params, 'lr': training_args.flame_pose_lr, "name": "pose"}
        # self.optimizer.add_param_group(param_pose)

        # # translation
        # self.flame_param['translation'].requires_grad = True
        # param_trans = {'params': [self.flame_param['translation']], 'lr': training_args.flame_trans_lr, "name": "trans"}
        # self.optimizer.add_param_group(param_trans)
        
        # # expression
        # self.flame_param['expr'].requires_grad = True
        # param_expr = {'params': [self.flame_param['expr']], 'lr': training_args.flame_expr_lr, "name": "expr"}
        # self.optimizer.add_param_group(param_expr)
        ###########################################################################

    def save_ply(self, path):
        super().save_ply(path)

        npz_path = Path(path).parent / "flame_param.npz"
        flame_param = {k: v.cpu().numpy() for k, v in self.flame_param.items()}
        np.savez(str(npz_path), **flame_param)

        if self.dynamic == 'FeatureLine':
            torch.save(self.feat_line.state_dict(), Path(path).parent / "feat_line.pth")
            # torch.save(self.feat_line_face.state_dict(), Path(path).parent / "feat_line_face.pth")
            # torch.save(self.feat_line_head.state_dict(), Path(path).parent / "feat_line_head.pth")
        if self.static == 'triplane':
            torch.save(self.triplane.state_dict(), Path(path).parent / "triplane.pth")

    def load_ply(self, path, **kwargs):
        super().load_ply(path)
        if self.dynamic == 'FeatureLine':
            self.feat_line.load_state_dict(torch.load(Path(path).parent / "feat_line.pth"))
            # self.feat_line_face.load_state_dict(torch.load(Path(path).parent / "feat_line_face.pth"))
            # self.feat_line_head.load_state_dict(torch.load(Path(path).parent / "feat_line_head.pth"))
        if self.static == 'triplane':
            self.triplane.load_state_dict(torch.load(Path(path).parent / "triplane.pth"))

        if not kwargs['has_target']:
            # When there is no target motion specified, use the finetuned FLAME parameters.
            # This operation overwrites the FLAME parameters loaded from the dataset.
            npz_path = Path(path).parent / "flame_param.npz"
            flame_param = np.load(str(npz_path))
            flame_param = {k: torch.from_numpy(v).cuda() for k, v in flame_param.items()}

            self.flame_param = flame_param
            self.num_timesteps = self.flame_param['expr'].shape[0]  # required by viewers
        
        if 'motion_path' in kwargs and kwargs['motion_path'] is not None:
            # When there is a motion sequence specified, load only dynamic parameters.
            motion_path = Path(kwargs['motion_path'])
            flame_param = np.load(str(motion_path))
            flame_param = {k: torch.from_numpy(v).cuda() for k, v in flame_param.items() if v.dtype == np.float32}

            self.flame_param['translation'] = flame_param['translation']
            self.flame_param['rotation'] = flame_param['rotation']
            self.flame_param['neck_pose'] = flame_param['neck_pose']
            self.flame_param['jaw_pose'] = flame_param['jaw_pose']
            self.flame_param['eyes_pose'] = flame_param['eyes_pose']
            self.flame_param['expr'] = flame_param['expr']
            self.flame_param['static_offset'] = flame_param['static_offset']
            self.flame_param['jaw_quat_weight'] = flame_param['jaw_quat_weight']
            self.flame_param['shape'] = flame_param['shape']

            self.num_timesteps = self.flame_param['expr'].shape[0]  # required by viewers
        
        if 'disable_fid' in kwargs and len(kwargs['disable_fid']) > 0:
            mask = (self.binding[:, None] != kwargs['disable_fid'][None, :]).all(-1)

            self.binding = self.binding[mask]
            self._xyz = self._xyz[mask]
            if self.static == 'None':
                self._features_dc = self._features_dc[mask]
                self._features_rest = self._features_rest[mask]
            self._scaling = self._scaling[mask]
            self._rotation = self._rotation[mask]
            self._opacity = self._opacity[mask]
