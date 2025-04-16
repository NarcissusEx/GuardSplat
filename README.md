# GuardSplat: Efficient and Robust Watermarking for 3D Gaussian Splatting

[Zixuan Chen](https://narcissusex.github.io), [Guangcong Wang](https://wanggcong.github.io/), [Jiahao Zhu](), [Jian-Huang Lai](https://cse.sysu.edu.cn/content/2498), [Xiaohua Xie](https://cse.sysu.edu.cn/content/2478). 

**IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)**, 2025

**TL;DR:** GuardSplat is an efficient watermarking framework to protect the copyright of 3DGS assets. It presents superior performance to state-of-the-art watermarking approaches in capacity, invisibility, robustness, security, and training efficiency.

---

<!-- [[Paper]](https://arxiv.org/abs/2411.19895) [[Project Page]](https://narcissusex.github.io/GuardSplat/) [[Demo Video]](https://youtu.be/QgejiJE2-5g) [[Materials & Pretrained Models]](https://drive.google.com/drive/folders/1U3OR5z5EOC7S5bicS2aYi199GX2Lv5e3?usp=drive_link) -->

<span class="links">
  <a  href="https://arxiv.org/abs/2411.19895" rel="nofollow"><img src="https://img.shields.io/badge/cs.CV-2411.19895-b31b1b?logo=arxiv&logoColor=red" alt="ArXiv" style="max-width: 100%;"></a>
  <a  href="https://arxiv.org/abs/2411.19895" rel="nofollow"><img src="https://img.shields.io/badge/Paper-gray?logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAADIAAAAuCAYAAABqK0pRAAAKAElEQVRo3u1aWW8b1xlNkfahQG0nLlwg6IIW%2FQnuH%2Bjv6PbsN%2F8Bo08BGsBLU%2BexNYq2iW35wW%2FKQ2HXjrWLnBmSM9xFiqS4WItJStRCUvLXc747Q116QUPJdRIgA1zM8M72Leec77sjvfPOt9u329drE5Fzg8HgV%2F3%2B8MN%2Bv%2F%2Fv%2Ff39eYy5%2F8c4ODiwfh%2Fo7z28r1Zb%2BytM%2Be6Xsfc7luHvDYfDD4%2BOjq5hf%2FXw8DCG47685Q3vxHguOzs9SSSTPkz7fmTgu3t7ez%2BDt7%2FgwO%2Bfc48bPsNYhsFL4Ui8%2BNDnz58L5k88aBSfYY9er6fz0dYfDGRzc2v0e2dnB9fsysbGJh1xRo7gxg9wvoybe7zOGq%2FcXnzxacb29rbu7Y1Gn8iRzc3NH%2FP%2Btw0ROvDs2TMejM0D99J%2B1j6xI%2BtfhSPl1ob0Bodj81P37slfbt4cZeVr7whckblSUxIbu2Ozt2%2FfkWvXr785R%2FggyKoOyKxAqWSAYwiBQP50zj7Pee55jse8fjgYKqlf50i60pBGbzA2e%2BfuXblx48abc6SHG2afzMjMkycyPzcnqURCEp4rszOz4jqOzM%2FOytLCoizML0iQ8vV4eXFR3HhcFjE3PzsnM198Ifl84bU5WcikpLHbfQlaN79Z0BL529ID%2BUdyRoZHx1m7OzUlf%2F7442%2BOIyR7o96QDhTKlltm5DqhFUKSMK2v1U%2FvCHnRB%2B6PDo%2F%2Bp2GRlNq1wi56L17f7nRees7tO3fk6rVrI%2Bd2d3elXF49vSOrpZL4yaS0251Xnl%2BrVCTjB7JSKEoRfPATSSkXV%2FS4kM3inC%2FZIC2ZVEqviQykI81mU3a622OCQGjd%2FOST0XV7J3GE0acRNRjXQDrXKlVZq%2FJ4DQ6VcVzDuVWkeg37qipTEca6yzGIQVISjivxpWXdJ3kM8jtLS%2BJ7niQgApynskUbBcTBvft7%2B6O5z27fHpPfvd09qeBdEznSqtf1hV4sLkEihegmNMJpRFMNiTuS8hL6O5%2FJquOrKyuYdzHnm%2FuSqdE9Pq51YzEp5PL6LF7T7x874uJ5HgaNHWXk7pRcvXp1DFrVSm0yR%2Bq1mr5MjcILGEkaQ1h4MMiDMw4i7iGKNHYXDywXCuJizofjSc2EI0nXU7glHU%2BfxWuZNTrJLEZbNpORJghv8%2Bdfn34qf%2Froo9NxhHgPwAcaTQOyQaCZyKbTmgkaWsjlJIffHP0Dk5GU6yqEUq7JVnxxCRkl1Bx9Du%2F1vaQeR3yg8a3WU9lB43hkcWR6%2BnOZmro3cgTrD2k0mpM50mBGksYADsLHXV4OjzNKWjpAmKQQdVbxarkM%2BMTVSe7TIHcSjtFwOkg4MosRLG2O%2BCykKJxs3aNtfX1jTEz2wB82khM7QgMZOcKDkGBklSswIpdJq8E0iJlje1IpGUey6axmw0DKHTnHZzCLhFALKmXXjET4Lps3efCpZpFboVWaEFp1qJLiGlhOqSMGTjSOGSHMdA6RJuzoSHtrSyNeTAdw0FO5zYSw5D3kBWW4BOmtAuuHwxBaGGlcYxzpjzmSSQWj7h6LPqlV1yYle1XhQJJn8JIgzIiSV7HuKf7VIWTlAPhdyefVEc57IXx4Xnmj1zo6oiwPwuiTIzSaQWP1HsENmc4AwpEAnKiO1KtVzURGlSulL1dsY87IakKdoaMGWgMpg%2BxUJAMjUxeU4K5nnMB81k9rQNLIrF38kuBRFJBoq6MEUACOObI3ufyy8KliwWhGj8ZEXKGRAWuIws5VR0n2Cio%2FocdhMrYMfjh6DdXO1ww5%2BgwWTZsPHrkEx21HSHY%2Bd8yRam3SgtiQfDqjLye%2BTVEzhEzhpSx4EUx4jvJL3LOuqDqxwuNeXsvaQ3hqMPCbGaWzdh3xwEfCct9yJAc%2B8doIWiciexMZSVCBgFHKrk3eSL0IKdYLSvMBsM3ao7DyonO4B4YUcZ7ZZIdAyWam%2BZzh8BhaTsw4b2eAvEkBcnZGVlcrkzmytbGh%2FVUTOGWzuI6CRXIy8uyAo9VgtGfUWEeYDWbL0Qqf1AxFnGLFZ5YiHkV1hPcWoGQ18HJgwS0DGSfZTwUtpp03RuPF1vtVWylsURj9hGsIz8hTxglPSjWdYj9Gp2xHctmc8shWLTaSrEnHqoWMlCfMyHa3i4hktOJS4zuddrj2HowGDTkM51jcWB9IbsKREWd2suACRYE84lzUv%2BXw7HGOuApFmyM%2BeZk45ghhN3GLQtnLM7UwpLxS0pcXtUD5JqqOUSIaSaMYyRWsO5Jh8eR50zUftzmEnPZZoRLyg8TIEa0x7hi0suAW1y12QaxWq5PKb00jSkciY0hsEl4rfbiucMKeSqMFgXAoDFo44UgqoU6bOpRSkWAg3OW4NBuNsRblVfL74pfGE5E9Wu0Fox7LqBCjTwXjcQERY3SDsCK3kHYaQ5Wj5JYKeYVbLmw%2FcuzBoGY53Ef42dBiQUxq89m3CmJDv%2BCMye%2BkHKECcYWXCfFuSOtqtKNGUesB5ukwi1sphFYqXIOktV4QkjnTTAbpkeO6HhnYjiSkgjpkd8ROLBbWkTAj%2B7tSqa5OLr%2BmFrhh45cO60BGI7%2BJmzfW1%2FW6TexpQLlY1FWgj5dzT55wAWZqUUyW5xcNb3RlGYyp1tOnL38qoMgwaFFGuAxm2zKxI37Y9JGkhAOrNAtaNmwiWSuYJf7mS7iWZ8T9MFtOWP1NXYlJsZCRUhXdby0nZYzBcGAtrFq6yrRlnv0Yl8a2%2FJ6oaYzW6TTGtBaGLwotQi0eD9sNX7G9Wlwx58JrymgxVDDYcEIslPi4lo5yXWITmS2MB%2Bm22%2FgglGqbI6snWepG3S4HW3MqD2ESW1jS4YVtOSOuioLud3WlrM2jccCsZeiYp%2BsXo1q694MxPiSpdAiWXUfIRyLBzsjElb0OR%2FywtfZ0TRGuP8IPCrGFRXWKqsTFEslu1vmmnyLEKBaUWha1IAAEs0G4RnFDslvdL%2BbieKb9OaiITqFeq4%2F9fWRjY8JPpr2dbf2uRTiZFZ4pjnkYzcJH%2FLaaLfRgLbxsTSHBjJg236xfeE8a6w8SO0iYjxHRh4tMgMpufXxgxW5vPRuT5An%2FYuW%2B9ksjH8IWhBBgxY1alGiuH35CZTHUvwVGLUx4Pvr7oN43HI4d252v%2FsWq3X5JtWig7dgAx93u8ULLNKuic3HHcd%2FIR%2BzDw%2BGpPmJ3ut2X5jud8TlmMJ8vVre2tgqwtZjNZpcfPPjP32%2FduvXHK1eu%2FBouvBs58hNc3%2F4q%2FqzQbLW6UKUmRosD4tFaWFj45%2BPHj3%2Fz6NGj3z98%2BPAP09PTv7148eIvz549e%2F7MmTM%2FhMlnX%2FmH9Gaz%2BaNer%2Feo0%2Bn43W439bYG33f%2F%2Fv3fXb58%2BYNLly79NBrnz7%2FG0C%2FzTwEXLlz4wblz597HeO8tjvfx7u%2B9qf%2FE%2BC%2FkEJZILUQcyQAAAABJRU5ErkJggg%3D%3D&label=CVPR%20(Coming%20Soon)&labelColor=367DBD" alt="CVPR" style="max-width: 100%;"></a>
  <a href="https://narcissusex.github.io/GuardSplat" rel="nofollow"><img src="https://img.shields.io/badge/Website-gray?logo=googlechrome&label=Project%20Page&labelColor=orange&logoColor=white" alt="Projectpage" style="max-width: 100%;"></a>
  <a href="https://youtu.be/QgejiJE2-5g" rel="nofollow"><img src="https://img.shields.io/badge/Demo%20Video-gray?logo=Youtube&logoColor=white&label=YouTube&labelColor=FF0000" alt="YouTube" style="max-width: 100%;"></a>
</span>


<div align=center>
<img width="1148" alt="framework" src="assets/guardsplat.png">
</div>
<b>Application scenarios of GuardSplat.</b> To protect the copyright of 3D Gaussian Splatting (3DGS) assets, <b>(a)</b> the owners (<font style="color:#058CFA;">Alice</font>) can use our <b>GuardSplat</b> to embed the secret message (<font style="color: blue;">blue key</font>) into these models. <b>(b)</b> If malicious users (<font style="color:#00B050">Bob</font>) render views for unauthorized uses, <b>(c)</b> <font style="color:#058CFA;">Alice</font> can use the private message decoder to extract messages (<font style="color:plum">purple key</font>) for copyright identification.

<!-- ## Abstract

3D Gaussian Splatting (3DGS) has recently created impressive assets for various applications. However, the copyright of these assets is not well protected as existing watermarking methods are not suited for 3DGS considering security, capacity, and invisibility. Besides, these methods often require hours or even days for optimization, limiting the application scenarios. In this paper, we propose **GuardSplat**, an innovative and efficient framework that effectively protects the copyright of 3DGS assets.
Specifically, **1)** We first propose a CLIP-guided Message Decoupling Optimization module for training the message decoder, leveraging CLIP's aligning capability and rich representations to achieve a high extraction accuracy with minimal optimization costs, presenting exceptional **capability** and **efficiency**.
**2)** Then, we propose a Spherical-harmonic-aware (SH-aware) Message Embedding module tailored for 3DGS, which employs a set of SH offsets to seamlessly embed the message into the SH features of each 3D Gaussian while maintaining the original 3D structure.
It enables the 3DGS assets to be watermarked with minimal fidelity trade-offs and prevents malicious users from removing the messages from the model files, meeting the demands for **invisibility** and **security**.
**3)** We further propose an Anti-distortion Message Extraction module to improve **robustness** against various visual distortions.
Extensive experiments demonstrate that **GuardSplat** outperforms the state-of-the-art methods and achieves fast optimization speed. -->

## Framework

<div align=center>
<img width="1148" alt="framework" src="assets/framework.png">
</div>

**Overview of GuardSplat.** **(a)** Given a binary message $M$, we first transform it into CLIP tokens $T$ using the proposed message tokenization.
We then employ CLIP's textual encoder $\mathcal{E _T}$ to map $T$ to the textual feature $F _\mathcal{T}$.
Finally, we feed $F _\mathcal{T}$ into message decoder $\mathcal{D _M}$ to extract the message $\hat{M}$ for optimization.
**(b)** For each 3D Gaussian, we freeze all the attributes and build a learnable spherical harmonic (SH) offset $\boldsymbol{h}^o _i$ as the watermarked SH feature, which can be added to the original SH features as $\boldsymbol{h} _i + \boldsymbol{h}^o _i$ to render the watermarked views.
**(c)** We first feed the 2D rendered views to CLIP's visual encoder $\mathcal{E _V}$ to acquire the visual feature $F _{\mathcal{V}}$ and then employ the pre-trained message decoder to extract the message $\hat{M}$.
A differentiable distortion layer is used to simulate various visual distortions during optimization.
$\mathcal{D _M}$ and $\boldsymbol{h}^o _i$ are optimized by the corresponding losses, respectively.

## 1) Get start

* Python 3.12
* CUDA 12.1 or *higher*
* NVIDIA RTX 3090
* PyTorch 2.5.1 or *higher*

**Create a python env using conda**
```bash
conda create -n GuardSplat python=3.12 -y
conda activate GuardSplat
```

**Install the required libraries**
```bash
bash setup.sh
```

Please see *setup.sh* in details.

## 2) Train and evaluate the message decoder
```bash
python make_decoder.py --mode train --msg_len <message_length> --save --num_epochs <training_epochs>
python make_decoder.py --mode test --msg_len <message_length>
```

## 3) Train a 3DGS model
```bash
python gaussian-splatting/train.py -s <nerf_dir>/<nerf_item> -m <result_dir> -w 
```

## 4) Watermark a 3DGS model (blender or llff mode)
```bash
# no distortions
python run_watermark.py -s <nerf_dir>/<nerf_item> -m <result_dir> -w --mode train --msg_len <message_length> --sdir <watermark_dir> --dtype blender

# single distortion
python run_watermark.py -s <nerf_dir>/<nerf_item> -m <result_dir> -w --mode train --msg_len <message_length> --sdir <watermark_dir> --dtype blender atypes <distortion>

# combined distortions
python run_watermark.py -s <nerf_dir>/<nerf_item> -m <result_dir> -w --mode train --msg_len <message_length> --sdir <watermark_dir> --dtype blender atypes <distortion1> <distortion2> ... <distortionN>
```

More details can be shown in *run.sh*.

## Citation

```tex
@InProceedings{chen2025guardsplat,
    author={Chen, Zixuan and Wang, Guangcong and Zhu, Jiahao and Lai, Jian-Huang and Xie, Xiaohua},
    title={GuardSplat: Efficient and Robust Watermarking for 3D Gaussian Splatting},
    year={2025},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision and Pattern Recognition (CVPR)},
}
```

## Acknowledgement 

We build our project based on **[gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)** and **[CLIP](https://github.com/openai/CLIP)**.
The differentiable JPEG compression and VAE attack are implemented based on **[Diff-JPEG](https://github.com/necla-ml/Diff-JPEG)** and **[WatermarkAttacker](https://github.com/XuandongZhao/WatermarkAttacker)**, respectively.
We also follow the settings used in **[CopyRNeRF](https://github.com/luo-ziyuan/CopyRNeRF-code/)** and **[WateRF](https://github.com/kuai-lab/cvpr2024_WateRF)**.
We sincerely thank them for their wonderful work and code release.