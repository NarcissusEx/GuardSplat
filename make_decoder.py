##########################################################################
# @Author: Zixuan Chen
# @Email:  chenzx3@mail2.sysu.edu.cn
# @Date: 2025-04-15
# @Description: Build message decoder for GaurdSplat.
# This Software is free for non-commercial, research and evaluation use.
##########################################################################

import os
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import trange

from wm_utils import CLIPWatermarker, MsgDataset, bit_accuracy

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def single_step_inference(batch, model, iscpu=False):
    input, target = [x.to(DEVICE) for x in batch]
    output = model.autoencoder(input)
    return (output.float().cpu(), target.cpu()) if iscpu else (output.float(), target)

def train(args):
    print (f'training decoder with N_L={args.msg_len}')
    dataset = MsgDataset(msg_len=args.msg_len, max_size=args.max_size)
    trainloader = DataLoader(dataset, batch_size=min(args.batch_size, dataset.__len__()), shuffle=True)
    testloader = DataLoader(dataset, batch_size=min(args.batch_size, dataset.__len__()), shuffle=False)
    model = CLIPWatermarker(msg_len=args.msg_len).to(DEVICE)
    optimizer = optim.Adam(model.msg_decoder.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    all_epochs = trange(1, args.num_epochs + 1)

    for eIdx in all_epochs:
        loss_value = 0
        for batch in trainloader:
            optimizer.zero_grad()
            loss = loss_fn(*single_step_inference(batch, model))
            loss.backward()
            optimizer.step()
            loss_value += loss.item()

        score = test(args, model, testloader)
        model.train()

        all_epochs.set_description(f'BAcc : {score} | Loss : {loss_value / dataset.__len__()}')

    if args.save:
        os.makedirs(args.sdir, exist_ok=True)
        sdict = {
            'model'   : model.eval().cpu().state_dict(),
            **{key : getattr(dataset, key) for key in ['data', 'b2t_maps']}
        }
        torch.save(sdict, os.path.join(args.sdir, f'CLIP-MsgDecoder-{args.msg_len}.pkl'))

def test(args, model=None, testloader=None):
    if model is None:
        print (f'testing decoder with N_L={args.msg_len}')
        sdict = torch.load(os.path.join(args.sdir, f'CLIP-MsgDecoder-{args.msg_len}.pkl'), map_location='cpu', weights_only=True)
        model = CLIPWatermarker(msg_len=args.msg_len)
        model.load_state_dict(sdict.pop('model'))
        model.to(DEVICE)
        dataset = MsgDataset(msg_len=args.msg_len, max_size=args.max_size, **sdict)
        testloader = DataLoader(dataset, batch_size=min(args.batch_size, dataset.__len__()), shuffle=False)

    with torch.no_grad():
        model.eval()
        outputs, targets = list(zip(*[single_step_inference(batch, model, iscpu=True) for batch in testloader]))
        outputs, targets = torch.cat(outputs), torch.cat(targets)

    if args.mode == 'train':
        return bit_accuracy(outputs, targets)
        
    else:
        print (f'BAcc : {bit_accuracy(outputs, targets):.2f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], required=True)
    parser.add_argument('--msg_len', type=int, choices=[4, 8, 16, 32, 48, 64, 72, 96, 128, 160, 192, 224, 256])
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--sdir', type=str, default='decoders')
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_size', type=int, default=2048)
    parser.add_argument('--save', action='store_true') # whether saving ckpt file
    args = parser.parse_args()

    globals()[args.mode](args)