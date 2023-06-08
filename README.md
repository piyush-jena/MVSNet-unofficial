# MVSNet: Depth Inference for Unstructured Multi-view Stereo
[MVSNet: Depth Inference for Unstructured Multi-view Stereo](https://arxiv.org/abs/1804.02505). Yao Yao, Zixin Luo, Shiwei Li, Tian Fang, Long Quan. ECCV 2018. 
MVSNet is a deep learning architecture for depth map inference from unstructured multi-view images.

This is an unofficial Pytorch implementation of MVSNet intended to be an entrant for Image Matching Challenge 2023 in Kaggle.

## How to Use

### Environment
* python 3.10 (Anaconda)
* pytorch 2.0.1
* CUDA 11.7

### Training

* Download the preprocessed [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view) (Fixed training cameras, from [Original MVSNet](https://github.com/YoYo000/MVSNet)), and upzip it as the ``MVS_TRANING`` folder
* create a logdir called ``checkpoints``
* Train MVSNet: ``python train.py --batch_size 4 --numdepth 64``

## License
[MIT](https://choosealicense.com/licenses/mit/)
