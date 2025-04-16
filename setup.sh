# create python env
# conda create -n GuardSplat python=3.12 -y
# conda activate GuardSplat

# install pytorch
conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# install 3DGS (see https://github.com/graphdeco-inria/gaussian-splatting in details)
# NOTE, due to the bugs raised in the latest 3DGS version (see https://github.com/graphdeco-inria/gaussian-splatting/issues/1163),
# our GuardSplat is built on the older 3DGS version (472689c) with the main branch of diff-gaussian-rasterization,
# Please switch the branch of diff-gaussian-rasterization or replace the 3DGS with the bug-free one.
pip install ./gaussian-splatting/submodules/diff-gaussian-rasterization
pip install ./gaussian-splatting/submodules/simple-knn
# pip install ./gaussian-splatting/submodules/fused-ssim

# install python libraries
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# install CLIP (see https://github.com/openai/CLIP in details)
pip install git+https://github.com/openai/CLIP.git

# install Diff-JPEG (see https://github.com/necla-ml/Diff-JPEG).
pip install git+https://github.com/necla-ml/Diff-JPEG