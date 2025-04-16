##########################################################################
# @Author: Zixuan Chen
# @Email:  chenzx3@mail2.sysu.edu.cn
# @Date: 2025-04-15
# @Description: Script of GaurdSplat.
# This Software is free for non-commercial, research and evaluation use.
##########################################################################

##########################################################################
# Train and evaluate message decoder
# msg_len | num_epochs
# 4 8 16 ->  100
# 32 48  ->  200
# 64 72  ->  300
# for example
python make_decoder.py --mode train --msg_len $msg_len --save --num_epochs $num_epochs
python make_decoder.py --mode test --msg_len $msg_len
##########################################################################

##########################################################################
# Train 3DGS model (see https://github.com/graphdeco-inria/gaussian-splatting in details)
# this script is an example for NeRF Synthetic data
root=<your_nerf_dir>
mdir=<your_result_dir>
item=<nerf_item>
python gaussian-splatting/train.py -s $root/$item -m $mdir -w
##########################################################################

##########################################################################
# Train and evaluate GuardSplat
# this script is an example for NeRF Synthetic data
root=<your_nerf_dir>/nerf/nerf_synthetic
mdir=<your_result_dir>
msg_len={4, 8, 16, 32, 48, 64, 72}
item=<nerf_item>
sdir=<your_watermark_dir>
# no_distortion
python run_watermark.py -s $root/$item -m $mdir -w --mode train --msg_len $msg_len --sdir $sdir --dtype blender
python run_watermark.py -s $root/$item -m $mdir -w --mode eval --msg_len $msg_len --sdir $sdir --dtype blender --save_map

# Visual distortions
# slightly increasing the value of lambda_msg, learning_rate, and num_epochs may achieve better bit accuracy against visual distortions.
# e.g., lambda_msg -> 0.04, learning_rate -> 0.01, num_epochs -> 150, etc.

## single distortion
atype=<distortion>
python run_watermark.py -s $root/$item -m $mdir -w --mode train --msg_len $msg_len --sdir $sdir --dtype blender --atypes $atype
python run_watermark.py -s $root/$item -m $mdir -w --mode eval --msg_len $msg_len --sdir $sdir --dtype blender --atypes $atype --save_map

## combined distortions
python run_watermark.py -s $root/$item -m $mdir -w --mode train --msg_len $msg_len --sdir $sdir --dtype blender --atypes <distortion1> <distortion2> ... <distortionN>
python run_watermark.py -s $root/$item -m $mdir -w --mode eval --msg_len $msg_len --sdir $sdir --dtype blender --atypes <distortion1> <distortion2> ... <distortionN> --save_map
##########################################################################