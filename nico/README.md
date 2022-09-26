# Image Classification on NICO

This is the codebase for the image classification experiments in our ICML 2022 paper [Fighting Fire with Fire: Avoiding DNN Shortcuts through Priming](https://proceedings.mlr.press/v162/wen22d/wen22d.pdf).
For the details of our method, please refer to our paper or [website](https://sites.google.com/view/icml22-fighting-fire-with-fire).

This codebase is built on the framework of [CaaM](https://github.com/Wangt-CN/CaaM).
The Key Input generation function is based on [BASNet](https://github.com/xuebinqin/BASNet). Thanks to the authors for providing these excellent codes.

### 1. Preparation

(1) Installation: Python3.6, Pytorch1.6, tensorboard, timm(0.3.4), scikit-learn, opencv-python, matplotlib, yaml, tqdm

(2) Dataset: 

- Please download the NICO dataset from the CAAM paper's link: https://drive.google.com/drive/folders/17-jl0fF9BxZupG75BtpOqJaB6dJ2Pv8O?usp=sharing.

(3) Generate Key Inputs for NICO dataset:
```
CUDA_VISIBLE_DEVICES=0 python generate_key_inputs.py --image-dir $path_of_NICO_images --output-folder $path_to_save_key_inputs
```

For the details about Key Input, please refer to the Section3.2 in our paper.

(4) Please remember to change the data path in the config file.


### 2. Evaluation:

For ResNet18 on NICO dataset

```
CUDA_VISIBLE_DEVICES=0 python train.py -cfg conf/primenet_resnet18_bf0.02.yaml -debug -gpu -eval pretrain_model/nico_primenet_resnet18-best.pth
```

The results will be:

- In-Domain Val Score: 0.7111111283302307

- Val Score: 0.513076901435852

- Test Score: 0.49000000953674316

The pretrained model is in `pretrain_model`, which is trained on a 3090 GPU.

### 3. Train

To perform training, please run the sh file in scripts. For example:

```
sh scripts/run_baseline_resnet18.sh
sh scripts/run_primenet_resnet18.sh
```

### Citation:

If you find this work is useful for your research, please cite our paper:

```
@inproceedings{wen2022fighting,
  title={Fighting Fire with Fire: Avoiding DNN Shortcuts through Priming},
  author={Wen, Chuan and Qian, Jianing and Lin, Jierui and Teng, Jiaye and Jayaraman, Dinesh and Gao, Yang},
  booktitle={International Conference on Machine Learning},
  pages={23723--23750},
  year={2022},
  organization={PMLR}
}
```
