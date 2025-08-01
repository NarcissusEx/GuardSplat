##########################################################################
# @Author: Zixuan Chen
# @Email:  chenzx3@mail2.sysu.edu.cn
# @Date: 2025-04-15
# @Description: Watermarking process for GaurdSplat.
# This Software is free for non-commercial, research and evaluation use.
##########################################################################

import os
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from copy import deepcopy
from functools import reduce

import torch
from torch import nn, optim
import torch.nn.functional as F

from wm_utils import load_decoder_and_message, bit_accuracy, lpips
from attack_utils import Attacker
from nerf_utils import get_data_infos, get_cameras

# gaussian-splatting utils
import sys
sys.path.append(os.path.join(os.getcwd(), 'gaussian-splatting'))
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from scene import GaussianModel
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, l2_loss, ssim
from utils.general_utils import safe_state

SH_KEYS = ['features_dc', 'features_rest']

def render_views(gaussians, cameras, renderArgs):
    pds = [torch.clamp(render(camera, gaussians, *renderArgs)['render'], 0.0, 1.0)[None] for camera in cameras]
    return torch.cat(pds)

def extract_message(pds, model, args):
    messages = [model(pds[idx * args.batch_size : (idx + 1) * args.batch_size]) for idx in range((len(pds) + args.batch_size - 1) // args.batch_size)]
    return torch.cat(messages)

def eval_extraction_accuracy(pds, message, model, args):
    output = extract_message(pds, model, args)
    target = message.repeat(len(pds), *[1 for _ in range(len(message.shape))])
    return bit_accuracy(output, target)

def eval_extraction_accuracy_under_attacks(pds, message, attacker, model, args):
    rets = {}
    for atype in args.atypes:
        torch.cuda.empty_cache()
        attacked_pds = torch.cat([model.resample(torch.clamp(attacker(pd, atype)[None], 0, 1)) for pd in pds])
        rets[atype] = eval_extraction_accuracy(attacked_pds, message, model, args)

    return rets

def eval_image_quality(pds, gts, bsize=8):
    return {
        'psnr' : reduce(lambda x1, x2 : x1 + x2, [psnr(pd[None], gt[None]).item() for pd, gt in zip(pds, gts)]) / len(pds),
        'ssim' : reduce(lambda x1, x2 : x1 + x2, [ssim(pd[None], gt[None]).item() for pd, gt in zip(pds, gts)]) / len(pds),
        'lpips': reduce(lambda x1, x2 : x1 + x2, [lpips(pd[None], gt[None]).item() for pd, gt in zip(pds, gts)]) / len(pds)
    }

def train(dataset, opt, pipe, args, model, message):
    message_text = reduce(lambda x1, x2 : x1 + x2, [str(x) for x in message.int().cpu().numpy()])
    print (f'message : {message_text}')

    os.makedirs(args.sdir, exist_ok=True)
    with open(os.path.join(args.sdir, 'message.txt'), 'w') as txtfile:
        txtfile.write(message_text)
        print (f"message can be seen in {os.path.join(args.sdir, 'message.txt')}")

    gaussians, guardsplat = GaussianModel(dataset.sh_degree), GaussianModel(dataset.sh_degree)
    gaussians.load_ply(os.path.join(dataset.model_path, 'point_cloud', f'iteration_{args.checkpoint_num}', 'point_cloud.ply'))
    guardsplat.load_ply(os.path.join(dataset.model_path, 'point_cloud', f'iteration_{args.checkpoint_num}', 'point_cloud.ply'))

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    data_infos = get_data_infos(dataset.model_path, dataset.source_path, args.dtype)
    TestCameras = get_cameras(args.test_cams, args.dtype, deepcopy(data_infos), is_random=False)

    with torch.no_grad():
        gt_test = render_views(gaussians, TestCameras, (pipe, background))

    sh_features = {key : getattr(guardsplat, f'_{key}').clone() for key in SH_KEYS}
    sh_offsets = {key : nn.Parameter(torch.zeros_like(sh_feature).cuda().requires_grad_(True)) for key, sh_feature in sh_features.items()}
    params = [{'params' : [sh_offsets[key]], 'lr' : lr} for key, lr in zip(SH_KEYS, [args.learning_rate, args.learning_rate / args.k_rest])]
    optimizer = optim.Adam(params, lr=0.0, eps=1e-15)
    train_attacker, test_attacker = [Attacker(args.atypes, flag) for flag in [True, False]] if args.atypes is not None else [None, None]

    evaluation(guardsplat, None, TestCameras, test_attacker, (pipe, background), model, message, gt_test, 0, args.eval_interval, args)

    progress_bar = tqdm(range(1, args.num_epochs + 1), desc='Training progress')
    for iteration in progress_bar:
        iter_start.record()
        TrainCameras = get_cameras(args.train_cams, args.dtype, deepcopy(data_infos))        
        ldict, mdict = {k : [] for k in ['recon', 'msg', 'off']}, {k : [] for k in ['output', 'target']}
        for iterIdx in range((len(TrainCameras) + args.batch_size - 1) // args.batch_size):
            losses = {}
            optimizer.zero_grad(set_to_none=True)
            cameras = TrainCameras[iterIdx * args.batch_size : (iterIdx + 1) * args.batch_size]
            [setattr(guardsplat, f'_{key}', sh_features[key] + sh_offsets[key]) for key in SH_KEYS]

            pds = render_views(guardsplat, cameras, (pipe, background))
            with torch.no_grad():
                gts = render_views(gaussians, cameras, (pipe, background))
            
            # reconstruction loss
            rgb_loss = (1.0 - opt.lambda_dssim) * l1_loss(pds, gts) + opt.lambda_dssim * (1.0 - ssim(pds, gts))
            lpips_loss = lpips(pds, gts) / args.batch_size
            losses['recon'] = rgb_loss + lpips_loss

            if args.atypes is not None:
                attacked_pds = []
                for pd in pds:
                    atype = random.choice(['clean', *args.atypes])
                    attacked_pd = pd if atype == 'clean' else train_attacker(pd, atype)
                    attacked_pds.append(model.resample(attacked_pd[None]))
                output = model(torch.cat(attacked_pds))

            else:
                output = model(pds)

            target = message.repeat(len(output), *[1 for _ in range(len(message.shape))])
            
            # message and offset loss
            losses['msg'] = F.binary_cross_entropy_with_logits(output, target)
            losses['off'] = reduce(lambda x1, x2 : x1 + x2, [l2_loss(v, torch.zeros_like(v)) for _, v in sh_offsets.items()])
            loss = reduce(lambda x1, x2 : x1 + x2, [getattr(args, f'lambda_{k}') * v for k, v in losses.items()])
            loss.backward()
            optimizer.step()

            for k, v in losses.items():
                ldict[k].append(v.item())

            for k in mdict.keys():
                mdict[k].append(locals()[k])

        mean = lambda xs : sum(xs) / len(xs)
        ldict = {k : mean(v) for k, v in ldict.items()}
        mdict = {k : torch.cat(v) for k, v in mdict.items()}
        ans = bit_accuracy(**mdict)

        logs = reduce(lambda x1, x2 : f'{x1} | {x2}', [f'{k} : {v:.5f}' for k, v in ldict.items()])

        progress_bar.set_description(f'[ITER {iteration}] {logs} | Bit Acc : {ans:.4f}')
        iter_end.record()

        evaluation(guardsplat, sh_offsets, TestCameras, test_attacker, (pipe, background), model, message, gt_test, iteration, args.eval_interval, args)

@torch.no_grad()
def evaluation(gaussians, sh_offsets, cameras, attacker, renderArgs, model, message, gts, iteration, eval_interval, args):
    if iteration % eval_interval == 0 or args.mode == 'eval':
        pds = render_views(gaussians, cameras, renderArgs)

        ans = {
            'Bit Acc' : eval_extraction_accuracy(pds, message, model, args),
            **eval_image_quality(pds, gts)
        }

        if args.atypes is not None:
            ans.update(eval_extraction_accuracy_under_attacks(pds, message, attacker, model, args))
            
        logs = reduce(lambda x1, x2 : f'{x1} | {x2}', [f'{k} : {v:.4f}' for k, v in ans.items()])

        print (f'[ITER {iteration}] {logs}' if args.mode == 'train' else logs)

        if sh_offsets is not None and args.mode == 'train':
            torch.save({
                'message' : message,
                **sh_offsets
            }, os.path.join(args.sdir,  f'sh_offset_{str(iteration).zfill(len(str(args.num_epochs)))}.pkl'))
        torch.cuda.empty_cache()

        if args.save_map and args.mode == 'eval':
            rdir = os.path.join(args.sdir, 'views')
            print (f'Saving images in {rdir}')
            os.makedirs(rdir, exist_ok=True)
            pds = render_views(gaussians, cameras, renderArgs)
            zero_nums = len(str(len(pds)))
            pairs = zip(np.uint8(pds.permute(0, 2, 3, 1).cpu().numpy() * 255.), np.uint8(gts.permute(0, 2, 3, 1).cpu().numpy() * 255.))
            for idx, (pd, gt) in enumerate(pairs):
                Image.fromarray(pd).save(os.path.join(rdir, f'{str(idx).zfill(zero_nums)}_ours.png'))
                Image.fromarray(gt).save(os.path.join(rdir, f'{str(idx).zfill(zero_nums)}_gt.png'))
                
@torch.no_grad()
def eval(dataset, opt, pipe, args, model, *kwargs):
    gaussians, guardsplat = GaussianModel(dataset.sh_degree), GaussianModel(dataset.sh_degree)
    gaussians.load_ply(os.path.join(dataset.model_path, 'point_cloud', f'iteration_{args.checkpoint_num}', 'point_cloud.ply'))
    guardsplat.load_ply(os.path.join(dataset.model_path, 'point_cloud', f'iteration_{args.checkpoint_num}', 'point_cloud.ply'))

    with open(os.path.join(args.sdir, 'message.txt'), 'r') as txtfile:
        message_text = txtfile.read()
        message = torch.tensor([int(bit) for bit in message_text]).cuda()
        print (f'message : {message_text}')

    sh_features = {key : getattr(guardsplat, f'_{key}').clone() for key in SH_KEYS}
    filename = sorted(filter(lambda x : x.endswith('.pkl'), os.listdir(args.sdir)))[-1] if args.resume_num is None else f'sh_offset_{str(args.resume_num).zfill(len(str(args.num_epochs)))}.pkl'
    sh_offsets = torch.load(os.path.join(args.sdir, filename), weights_only=True, map_location='cuda')
    [setattr(guardsplat, f'_{key}', sh_features[key] + sh_offsets[key]) for key in SH_KEYS]
    print (f'Resuming from {os.path.join(args.sdir, filename)}')

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')

    data_infos = get_data_infos(dataset.model_path, dataset.source_path, args.dtype)
    TestCameras = get_cameras(args.test_cams, args.dtype, deepcopy(data_infos), is_random=False)

    gts = render_views(gaussians, TestCameras, (pipe, background))
    test_attacker = Attacker(args.atypes, False)

    evaluation(guardsplat, sh_offsets, TestCameras, test_attacker, (pipe, background), model, message, gts, 0, args.eval_interval, args)

if __name__ == '__main__':
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Training script parameters')
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--mode', choices=['train', 'eval'], required=True)
    parser.add_argument('--msg_len', type=int, choices=[4, 8, 16, 32, 48, 64, 72])
    parser.add_argument('--checkpoint_num', type=int, default=30000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--atypes', type=str, nargs='+')
    parser.add_argument('--sdir', type=str)
    parser.add_argument('--decoder_path', type=str, default='decoders')
    parser.add_argument('--test_cams', type=int, default=200)
    parser.add_argument('--train_cams', type=int, default=200)
    parser.add_argument('--dtype', choices=['blender', 'llff'])
    parser.add_argument('--quiet', action='store_true')

    # watermarking args
    parser.add_argument('--learning_rate', type=float, default=5e-3)
    parser.add_argument('--k_rest', type=float, default=20)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--lambda_msg', type=float, default=0.03)
    parser.add_argument('--lambda_off', type=float, default=10)
    parser.add_argument('--lambda_recon', type=float, default=1)
    
    # evaluation args
    parser.add_argument('--resume_num', type=int, default=None)
    parser.add_argument('--save_map', action='store_true')
    args = parser.parse_args(sys.argv[1:])
    
    if args.mode == 'train':
        pstr = 'Watermarking'
        print (f'Watermarking {args.model_path} with N_L={args.msg_len}')
        print (f'GuardSplat will be saved in {args.sdir}')

    else:
        pstr = 'Evaluation'
        print (f'Evaluating GuardSplat in {args.sdir}')

    # randomly choose a message
    wm_params = load_decoder_and_message(args.msg_len, args.decoder_path)
    safe_state(args.quiet)
    globals()[args.mode](lp.extract(args), op.extract(args), pp.extract(args), args, *wm_params)
    print(f'\n{pstr} complete.')