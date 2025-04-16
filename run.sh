##########################################################################
# @Author: Zixuan Chen
# @Email:  chenzx3@mail2.sysu.edu.cn
# @Date: 2025-04-15
# @Description: Script of GaurdSplat.
# This Software is free for non-commercial, research and evaluation use.
##########################################################################

##########################################################################
# train and evaluate message decoder
# msg_len | num_epochs
# 4 8 16 ->  100
# 32 48  ->  200
# 64 72  ->  300
# for example
msg_len=16
num_epochs=100
python make_decoder.py --mode train --msg_len $msg_len --save --num_epochs $num_epochs
python make_decoder.py --mode test --msg_len $msg_len
##########################################################################

##########################################################################
# train 3DGS model (see https://github.com/graphdeco-inria/gaussian-splatting in details)
# this script is an example for NeRF Synthetic data
root=<your_nerf_dir>/nerf/nerf_synthetic
mdir=<your_result_dir>
for item in lego chair drums ficus hotdog materials mic ship
do
    python gaussian-splatting/train.py -s $root/$item -m $mdir/$item -w 
done
##########################################################################

##########################################################################
# train and evaluate GuardSplat
# this script is an example for NeRF Synthetic data
root=<your_nerf_dir>/nerf/nerf_synthetic
mdir=<your_result_dir>
msg_len={4, 8, 16, 32, 48, 64, 72}

# no_distortion
distortion=no_distortion
for item in lego chair drums ficus hotdog materials mic ship
do
    python run_watermark.py -s $root/$item -m $mdir/$item -w --mode train --msg_len $msg_len --sdir $mdir/$item/N_L=$msg_len/$distortion --dtype blender
    python run_watermark.py -s $root/$item -m $mdir/$item -w --mode eval --msg_len $msg_len --sdir $mdir/$item/N_L=$msg_len/$distortion --dtype blender --save_map
done

# visual distortions
# increasing the value of lambda_msg, learning_rate, and num_epochs may achieve better bit accuracy against visual distortions.
## single distortion
for item in lego chair drums ficus hotdog materials mic ship
do
    for atype in GaussianBlur GaussianNoise Brightness JPEGCompress Crop Rotation Resize VAEBmshj2018
    do
        distortion=$atype
        python run_watermark.py -s $root/$item -m $mdir/$item -w --mode train --msg_len $msg_len --sdir $mdir/$item/N_L=$msg_len/$distortion --dtype blender --atypes $atype --lambda_msg 0.04 --learning_rate 0.01 --num_epochs 150
    done
done

## multi distortions
for item in lego chair drums ficus hotdog materials mic ship
do
    distortion=combined
    python run_watermark.py -s $root/$item -m $mdir/$item -w --mode train --msg_len $msg_len --sdir $mdir/$item/N_L=$msg_len/$distortion --dtype blender --atypes Crop JPEGCompress Brightness --lambda_msg 0.04 --learning_rate 0.01 --num_epochs 150
done
##########################################################################