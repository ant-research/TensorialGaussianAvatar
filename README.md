# 3D Gaussian Head Avatars with Expressive Dynamic Appearances by Compact Tensorial Representations(CVPR2025)

Yating Wang<sup>1</sup>, [Xuan Wang](https://xuanwangvc.github.io/)<sup>2</sup>, [Ran Yi](https://yiranran.github.io/)<sup>1</sup>, [Yanbo Fan](https://sites.google.com/site/yanbofan0124/)<sup>2</sup>, Jichen Hu<sup>1</sup>, Jingcheng Zhu<sup>1</sup>, [Lizhuang Ma](https://dmcv.sjtu.edu.cn/)<sup>1</sup>

Shanghai Jiaotong University<sup>1</sup>, AntGroup Research<sup>2</sup>

[arxiv](https://arxiv.org/abs/2504.14967)

This is the official implementation of the paper "3D Gaussian Head Avatars with Expressive Dynamic Appearances by Compact Tensorial Representations"

## Install

1. clone this repo
   
   `git clone  --recursive`
   
2. install cuda and pytorch
	
	Our experiments are conducted with cuda11.6, pytorch1.13.0, torchvision0.14.0

3. install requirements
   
	`pip install -r requirements.txt`

---

## Dataset and Preprocessing

### FLAME Model
Our method relies on FLAME face prior model(2023 version). Please download FLAME assets from [flame project](https://flame.is.tue.mpg.de/index.html), put flame2023.pkl(versions w/ jaw rotation) to flame_model/assets/flame/flame2023.pkl and put FLAME_masks.pkl to flame_model/assets/flame/FLAME_masks.pkl

### Test Data
We test our method on [NeRSemble](https://github.com/tobias-kirschstein/nersemble) multi-view human head videos dataset, which is preprocessed by [GaussianAvatars(CVPR2024)](https://github.com/ShenhanQian/GaussianAvatars/tree/main), please refers to [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars/blob/main/doc/download.md) to download test data. Unlike GaussianAvatars, we use free performance sequences as the test set and other video segments as the training set.

## Usage
### Preprocess

python 

### Training

`./run.sh`

### Rendering

`./render.sh`


## Acknowledgments

This work was heavily inspired by [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars/tree/main). We use [NeRSemble](https://github.com/tobias-kirschstein/nersemble) for testing. We also borrow code from the following repositories. Thanks to their impressive work!

1. Gaussian Splatting: https://github.com/graphdeco-inria/gaussian-splatting
   
2. Tri-planes: https://github.com/chiehwangs/gaussian-head

3. Axis Angle to Quanternion: https://lizhe00.github.io/projects/posevocab/
	

## Cite
If you find our paper or code useful in your research, please cite us with the following BibTeX:

`@misc{wang20253dgaussianheadavatars,
      title={3D Gaussian Head Avatars with Expressive Dynamic Appearances by Compact Tensorial Representations}, 
      author={Yating Wang and Xuan Wang and Ran Yi and Yanbo Fan and Jichen Hu and Jingcheng Zhu and Lizhuang Ma},
      year={2025},
      eprint={2504.14967},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.14967}, 
}`

