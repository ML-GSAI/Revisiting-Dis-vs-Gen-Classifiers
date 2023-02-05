# Revisiting Discriminative vs. Generative Classifiers: Theory and Implications

This is the official implementation for **Revisiting Discriminative vs. Generative Classifiers: Theory and Implications**.

## Dependencies

```bash
conda env create -f gen_vs_dis.yaml
```

## Simulation Experiments

```bash
python data/generate_data.py
bash scripts/main_simulation.sh
```

## Deep Learning Experiments

### Source of Pre-trained models

* ViT: checkpoint given by Google. 

  ```bash
wget https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz
  ```

* ResNet: pre-trained ResNet50 supported by Pytorch.

* CLIP: pre-trained checkpoint (backbone is ResNet50) supported by OpenAI ([link](https://github.com/openai/CLIP)).

* MoCov2: checkpoint given by FAIR ([link](https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar)).

* SimCLRv2: checkpoint given by Google ([link](https://console.cloud.google.com/storage/browser/simclr-checkpoints/simclrv2/pretrained/r50_1x_sk1)). The tenorflow checkpoint can be converted to Pytorch version by using the codes in https://github.com/Separius/SimCLRv2-Pytorch.

* MAE: checkpoint provided by FAIR ([link](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth)).

* SimMIM: checkpoint given by MSRA ([link](https://drive.google.com/file/d/1dJn6GYkwMIcoP3zqOEyW1_iQfpBi8UOw/view?usp=sharing)).

### Extract features on CIFAR10/CIFAR100

For example, when dataset is CIFAR10 and method is MoCov2, we can run

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PYTHONPATH:`pwd`
python main_extract_features.py \
  --dataset cifar10 \
  --backbone moco_v2 \
  --bs 100 \
  --gpu 1 \
```

### Analysis features

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PYTHONPATH:`pwd`
backbone_list=("clip" "resnet" "vit" "moco_v2" "simclr_v2" "mae" "simmim")
for backbone in ${backbone_list[@]};do
python plot.py \
  --dataset cifar10 \
  --backbone $backbone \
  --mode sigmas
done

backbone_list=("clip" "resnet" "vit" "moco_v2" "simclr_v2" "mae" "simmim")
for backbone in ${backbone_list[@]};do
python plot.py \
  --dataset cifar10 \
  --backbone $backbone \
  --mode kl
done

backbone_list=("clip" "resnet" "vit" "moco_v2" "simclr_v2" "mae" "simmim")
for backbone in ${backbone_list[@]};do
python plot.py \
  --dataset cifar10 \
  --backbone $backbone \
  --mode var_likelihood_diff
done
```

### Compare logistic regression and naive Bayes on the extracted features

For example, when dataset is CIFAR10 and method is MoCov2, we can run

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PYTHONPATH:`pwd`
python main_train_offline.py \
  --dataset cifar10 \
  --backbone moco_v2 \
  --model lr_bgfs \
  --C 1 \
  --repeat 5 \
  --minmax
```

## Hyperparameters Configuration

Detailed hyperparameters config can be found in scripts/main_plot.sh.

## Acknowledgments

The code is developed based on the following repositories. We appreciate their nice implementations.

| Method                           | Repository                                            |
| -------------------------------- | ----------------------------------------------------- |
| ViT                              | https://github.com/google-research/vision_transformer |
| ResNet                           | https://github.com/pytorch/pytorch                    |
| CLIP                             | https://github.com/openai/CLIP                        |
| MoCo_v2                          | https://github.com/facebookresearch/moco              |
| SimCLR_v2                        | https://github.com/google-research/simclr             |
| SimCLR_v2                        | https://github.com/Separius/SimCLRv2-Pytorch          |
| MAE                              | https://github.com/facebookresearch/mae               |
| SimMIM                           | https://github.com/microsoft/SimMIM                   |
| logistic regression, naive Bayes | https://github.com/scikit-learn/scikit-learn          |