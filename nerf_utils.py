##########################################################################
# @Author: Zixuan Chen
# @Email:  chenzx3@mail2.sysu.edu.cn
# @Date: 2025-04-15
# @Description: Generate the cameras of Blender and LLFF for watermarking.
# The code is implemented based on CopyRNeRF (https://github.com/luo-ziyuan/CopyRNeRF-code/)
# This Software is free for non-commercial, research and evaluation use.
##########################################################################

import os
import json
import torch
import numpy as np
from scene.cameras import MiniCam
from utils.graphics_utils import focal2fov, getProjectionMatrix, fov2focal
from PIL import Image

normalize = lambda x : x / torch.linalg.norm(x)

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def get_SE3_from_SO3_and_Trans(r, t):
    SE3 = torch.eye(4).repeat(len(r), 1, 1).cuda()
    SE3[:, :3, :3] = r if len(r.shape) == 3 else r.reshape(-1, 3, 3)
    SE3[:, :3, 3] = t
    return SE3

def get_SE3_from_m3x4(m3x4):
    r, t = m3x4[:, :3, :3], m3x4[:, :3, 3]
    return get_SE3_from_SO3_and_Trans(r, t)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec0 = normalize(torch.linalg.cross(up, vec2))
    vec1 = normalize(torch.linalg.cross(vec2, vec0))
    m = torch.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(c2ws):
    center = c2ws[:, :3, 3].mean(0)
    vec2 = normalize(c2ws[:, :3, 2].sum(0))
    up = c2ws[:, :3, 1].sum(0)
    return viewmatrix(vec2, up, center) #c2w

def recenter_poses(poses):
    poses_ = torch.clone(poses)
    bottom = torch.tensor([0, 0, 0, 1])[None].cuda()
    c2w = poses_avg(poses)
    c2w = torch.cat([c2w, bottom])
    poses = torch.cat([poses[:, :3, :4], bottom.repeat(len(poses), 1, 1)], 1)
    poses = torch.linalg.inv(c2w).repeat(len(poses), 1, 1).bmm(poses)
    poses_[:, :3, :4] = poses[:, :3, :4]
    return poses_

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(
        np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def get_data_infos(model_path, source_path, data_type):
    with open(os.path.join(model_path, f'cameras.json')) as jf:
        clist = json.load(jf)

    width = clist[0]['width']
    height = clist[0]['height']
    fovx = focal2fov(clist[0]['fx'], width)
    fovy = focal2fov(clist[0]['fy'], height)
    znear = 0.01
    zfar = 100.0

    if data_type == 'blender':
        return {
            'width' : width,
            'height': height,
            'fovx'  : fovx,
            'fovy'  : fovy,
            'znear' : znear,
            'zfar'  : zfar
        }

    else:
        Rs = torch.cat([torch.tensor(c['rotation'])[None] for c in clist]).cuda()
        Ts = torch.cat([torch.tensor(c['position'])[None] for c in clist]).cuda()
        w2cs = get_SE3_from_SO3_and_Trans(Rs, Ts)
        c2ws = torch.cat([torch.linalg.inv(w2c)[None] for w2c in w2cs])

        poses_arr = np.load(os.path.join(source_path, 'poses_bounds.npy'))
        bds = poses_arr[:, -2:].transpose([1, 0])
        bds = np.moveaxis(bds, -1, 0).astype(np.float32)
        bd_factor = 0.75
        sc = 1. if bd_factor is None else 1. / (bds.min() * bd_factor)
        bds *= sc
        close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
        c2ws[:, :3, 3] *= sc

        c2ws = recenter_poses(c2ws)
        c2w = poses_avg(c2ws)
        up = normalize(c2ws[:, :3, 1].sum(0))
        tt = c2ws[:, :3, 3]
        rads = torch.ones(4).cuda()
        rads[:3] = torch.quantile(torch.abs(tt), 0.9, 0)
        dt = 0.75
        focal = 1. / (((1. - dt) / close_depth + dt / inf_depth))
        
        return {
            'width' : width // 8,
            'height': height // 8,
            'fovx'  : fovx,
            'fovy'  : fovy,
            'znear' : znear,
            'zfar'  : zfar,
            'c2w'   : c2w,
            'up'    : up,
            'rads'  : rads,
            'focal' : float(focal)
        }
    
def get_supp_cameras(model_path, data_type):
    with open(os.path.join(model_path, f'cameras.json')) as jf:
        clist = json.load(jf)

    width = clist[0]['width']
    height = clist[0]['height']
    fovx = focal2fov(clist[0]['fx'], width)
    fovy = focal2fov(clist[0]['fy'], height)
    znear = 0.01
    zfar = 100.0

    params = {
        'width' : width if data_type == 'blender' else width // 8,
        'height': height if data_type == 'blender' else height // 8,
        'fovx'  : fovx,
        'fovy'  : fovy,
        'znear' : znear,
        'zfar'  : zfar
    }

    w2cs = []
    for c in clist:
        c2w = np.eye(4)
        rot = np.array(c['rotation'])
        pos = np.array(c['position'])
        c2w[:3, :3] = rot
        c2w[:3, 3] = pos
        w2c = np.linalg.inv(c2w).transpose()
        w2cs.append(w2c)
    
    w2cs = torch.from_numpy(np.array(w2cs)).cuda().float()
    proj_matrix = getProjectionMatrix(znear=params['znear'], zfar=params['zfar'], fovX=params['fovx'], fovY=params['fovy']).transpose(0,1).cuda()
    projs = w2cs.bmm(proj_matrix.repeat(len(w2cs), 1, 1))
    return [MiniCam(world_view_transform=w2c, full_proj_transform=proj, **params) for w2c, proj in zip(w2cs, projs)]

def get_cameras(num_cameras, data_type, data_infos, is_random=True):
    w2cs, params = globals()[f'get_{data_type}_poses'](num_cameras, data_infos, is_random)
    proj_matrix = getProjectionMatrix(znear=params['znear'], zfar=params['zfar'], fovX=params['fovx'], fovY=params['fovy']).transpose(0,1).cuda()
    projs = w2cs.bmm(proj_matrix.repeat(len(w2cs), 1, 1))
    return [MiniCam(world_view_transform=w2c, full_proj_transform=proj, **params) for w2c, proj in zip(w2cs, projs)]

def get_blender_poses(num_cameras, data_infos, is_random, theta_min=-180., theta_max=180., phi_min=-30., phi_max=-30., radius_min=4.031128874, radius_max=4.031128874):
    thetas = np.random.uniform(theta_min, theta_max, num_cameras) if is_random else np.linspace(theta_min, theta_max, num_cameras)
    phis = np.random.uniform(phi_min, phi_max, num_cameras)
    radiuss = np.random.uniform(radius_min, radius_max, num_cameras)
    c2ws = torch.stack([pose_spherical(t, p, r) for t, p, r in zip(thetas, phis, radiuss)], 0)
    c2ws[:, :3, 1:3] *= -1
    return torch.cat([torch.linalg.inv(c2w).transpose(0, 1)[None].cuda() for c2w in c2ws]), data_infos

def get_llff_poses(num_cameras, data_infos, is_random):
    c2w, up, rads, focal = [data_infos.pop(x) for x in ['c2w', 'up', 'rads', 'focal']]
    c2w = c2w.repeat(num_cameras, 1, 1)
    
    rots, zrate = 2, 0.5
    if is_random:
        thetas = torch.rand(num_cameras) * 2 * np.pi * rots  

    else: 
        thetas = torch.linspace(0., 2 * np.pi * rots, num_cameras + 1)[:-1]
    
    xs = torch.cat([torch.tensor([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.])[None].cuda() * rads[None] for theta in thetas])
    cs = torch.bmm(c2w, xs[..., None]).squeeze(-1)
    fs = torch.tensor([0, 0, -focal, 1]).repeat(num_cameras, 1)[..., None].cuda()
    zs = cs - torch.bmm(c2w, fs).squeeze(-1)
    m3x4s = torch.cat([viewmatrix(normalize(z), up, c)[None] for z, c in zip(zs, cs)])
    return torch.cat([torch.linalg.inv(c2w)[None] for c2w in get_SE3_from_m3x4(m3x4s)]), data_infos