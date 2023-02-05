export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PYTHONPATH:`pwd`
python plot.py \
  --dataset cifar10 \
  --backbone clip \
  --lr_path ./log/offline/clip_minmax/cifar10/lr_bgfs/C0.1/loss.npy \
  --nb_diag_path ./log/offline/clip_minmax/cifar10/nb_diag/loss.npy \
  --pic_dir ./log/offline/clip_minmax/cifar10/ \
  --mode short \


export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PYTHONPATH:`pwd`
python plot.py \
  --dataset cifar100 \
  --backbone clip \
  --lr_path ./log/offline/clip_minmax/cifar100/lr_bgfs/C0.2/loss.npy \
  --nb_diag_path ./log/offline/clip_minmax/cifar100/nb_diag/loss.npy \
  --pic_dir ./log/offline/clip_minmax/cifar100/ \
  --mode short \



export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PYTHONPATH:`pwd`
python plot.py \
  --dataset cifar10 \
  --backbone resnet \
  --lr_path ./log/offline/resnet_minmax/cifar10/lr_bgfs/C0.2/loss.npy \
  --nb_diag_path ./log/offline/resnet_minmax/cifar10/nb_diag/loss.npy \
  --pic_dir ./log/offline/resnet_minmax/cifar10/ \
  --mode short \

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PYTHONPATH:`pwd`
python plot.py \
  --dataset cifar100 \
  --backbone resnet \
  --lr_path ./log/offline/resnet_minmax/cifar100/lr_bgfs/C0.2/loss.npy \
  --nb_diag_path ./log/offline/resnet_minmax/cifar100/nb_diag/loss.npy \
  --pic_dir ./log/offline/resnet_minmax/cifar100/ \
  --mode short \


export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PYTHONPATH:`pwd`
python plot.py \
  --dataset cifar10 \
  --backbone moco_v2 \
  --lr_path ./log/offline/moco_v2_minmax/cifar10/lr_bgfs/C1.0/loss.npy \
  --nb_diag_path ./log/offline/moco_v2_minmax/cifar10/nb_diag/loss.npy \
  --pic_dir ./log/offline/moco_v2_minmax/cifar10/ \
  --mode short \

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PYTHONPATH:`pwd`
python plot.py \
  --dataset cifar100 \
  --backbone moco_v2 \
  --lr_path ./log/offline/moco_v2_minmax/cifar100/lr_bgfs/C1.0/loss.npy \
  --nb_diag_path ./log/offline/moco_v2_minmax/cifar100/nb_diag/loss.npy \
  --pic_dir ./log/offline/moco_v2_minmax/cifar100/ \
  --mode short \


export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PYTHONPATH:`pwd`
python plot.py \
  --dataset cifar10 \
  --backbone vit \
  --lr_path ./log/offline/vit_minmax/cifar10/lr_bgfs/C0.1/loss.npy \
  --nb_diag_path ./log/offline/vit_minmax/cifar10/nb_diag/loss.npy \
  --pic_dir ./log/offline/vit_minmax/cifar10/ \
  --mode short \

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PYTHONPATH:`pwd`
python plot.py \
  --dataset cifar100 \
  --backbone vit \
  --lr_path ./log/offline/vit_minmax/cifar100/lr_bgfs/C0.1/loss.npy \
  --nb_diag_path ./log/offline/vit_minmax/cifar100/nb_diag/loss.npy \
  --pic_dir ./log/offline/vit_minmax/cifar100/ \
  --mode short \


export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PYTHONPATH:`pwd`
python plot.py \
  --dataset cifar10 \
  --backbone simclr_v2 \
  --lr_path ./log/offline/simclr_v2_minmax/cifar10/lr_bgfs/C1.0/loss.npy \
  --nb_diag_path ./log/offline/simclr_v2_minmax/cifar10/nb_diag/loss.npy \
  --pic_dir ./log/offline/simclr_v2_minmax/cifar10/ \
  --mode short \


export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PYTHONPATH:`pwd`
python plot.py \
  --dataset cifar100 \
  --backbone simclr_v2 \
  --lr_path ./log/offline/simclr_v2_minmax/cifar100/lr_bgfs/C1.0/loss.npy \
  --nb_diag_path ./log/offline/simclr_v2_minmax/cifar100/nb_diag/loss.npy \
  --pic_dir ./log/offline/simclr_v2_minmax/cifar100/ \
  --mode short \

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PYTHONPATH:`pwd`
python plot.py \
  --dataset cifar10 \
  --backbone mae \
  --lr_path ./log/offline/mae_minmax/cifar10/lr_bgfs/C5.0/loss.npy \
  --nb_diag_path ./log/offline/mae_minmax/cifar10/nb_diag/loss.npy \
  --pic_dir ./log/offline/mae_minmax/cifar10/ \
  --mode short \

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PYTHONPATH:`pwd`
python plot.py \
  --dataset cifar100 \
  --backbone mae \
  --lr_path ./log/offline/mae_minmax/cifar100/lr_bgfs/C5.0/loss.npy \
  --nb_diag_path ./log/offline/mae_minmax/cifar100/nb_diag/loss.npy \
  --pic_dir ./log/offline/mae_minmax/cifar100/ \
  --mode short \

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PYTHONPATH:`pwd`
python plot.py \
  --dataset cifar10 \
  --backbone simmim \
  --lr_path ./log/offline/simmim_minmax/cifar10/lr_bgfs/C20.0/loss.npy \
  --nb_diag_path ./log/offline/simmim_minmax/cifar10/nb_diag/loss.npy \
  --pic_dir ./log/offline/simmim_minmax/cifar10/ \
  --mode short \

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PYTHONPATH:`pwd`
python plot.py \
  --dataset cifar100 \
  --backbone simmim \
  --lr_path ./log/offline/simmim_minmax/cifar100/lr_bgfs/C20.0/loss.npy \
  --nb_diag_path ./log/offline/simmim_minmax/cifar100/nb_diag/loss.npy \
  --pic_dir ./log/offline/simmim_minmax/cifar100/ \
  --mode short \


export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PYTHONPATH:`pwd`
backbone_list=("clip" "resnet" "vit" "moco_v2" "simclr_v2" "mae" "simmim")
for backbone in ${backbone_list[@]};do
python plot.py \
  --dataset cifar10 \
  --backbone $backbone \
  --mode sigmas
done


export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PYTHONPATH:`pwd`
backbone_list=("clip" "resnet" "vit" "moco_v2" "simclr_v2" "mae" "simmim")
for backbone in ${backbone_list[@]};do
python plot.py \
  --dataset cifar10 \
  --backbone $backbone \
  --mode kl
done


export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PYTHONPATH:`pwd`
backbone_list=("clip" "resnet" "vit" "moco_v2" "simclr_v2" "mae" "simmim")
for backbone in ${backbone_list[@]};do
python plot.py \
  --dataset cifar10 \
  --backbone $backbone \
  --mode var_likelihood_diff
done