import argparse
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import models.moco.backbones as backbones
from CLIP import clip
from models.moco.backbones import resnet50
import timm
from models.simclr_v2 import get_resnet, name_to_params
from models.simclr_v1 import resnet50x1
from models.mae import vit_base_patch16
from models.simmim import build_vit_B_16
from utils.tools import *


model_names = sorted(name for name in backbones.__dict__
    if name.islower() and not name.startswith("__")
    and callable(backbones.__dict__[name]))

parser = argparse.ArgumentParser(description='Extract features')
parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10','cifar100'],
                    help='dataset the features belong to.')
parser.add_argument('--backbone', default='clip', type=str, choices=['moco_v1', 'moco_v2', 'clip','resnet',
                    'vit', 'simclr_v2', 'simclr_v1', 'vit2', 'mae', 'simmim'],
                    help='pretrained backbone.')
parser.add_argument('--arch', metavar='ARCH', default='resnet50', choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('--bs', default=100, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--aug', dest='aug', action='store_true',
                    help='train lr with data augmentation')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

def main():
    args = parser.parse_args()
    args.device = torch.device('cuda', args.gpu) if args.gpu is not None else 'cuda'
    cudnn.benchmark = False
    cudnn.deterministic = True
    cudnn.enabled = False
    args.features_dir = os.path.join('./datasets', args.backbone, args.dataset)
    if not os.path.exists(args.features_dir):
        os.makedirs(args.features_dir)

    data_dir = os.path.join('./datasets')

    if args.aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Resize((224,224), interpolation=BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((224,224), interpolation=BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    else:
        if args.backbone not in ['simclr_v1', 'simclr_v2', 'vit']:
            transform_train = transform_test = transforms.Compose([
                transforms.Resize((224,224), interpolation=BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        else:
            transform_train = transform_test = transforms.Compose([
                transforms.Resize((224,224), interpolation=BICUBIC),
                transforms.ToTensor(),
            ])

    if args.dataset == 'cifar10':
        train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    else:
        train_set = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
        test_set = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.bs, shuffle=True)

    if args.backbone == 'moco_v1':
        args.backbone_path = './checkpoints/moco_v1_200ep_pretrain.pth.tar'
    elif args.backbone == 'moco_v2':
        args.backbone_path = './checkpoints/moco_v2_800ep_pretrain.pth.tar'
    elif args.backbone == 'clip':
        args.backbone_path = './checkpoints'
    elif args.backbone == 'simclr_v1':
        args.backbone_path = "./checkpoints/resnet50-1x.pth"
    elif args.backbone == 'simclr_v2':
        args.backbone_path = './checkpoints/r50_1x_sk1.pth'
    elif args.backbone == 'vit':
        args.backbone_path = "./checkpoints/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz"
    elif args.backbone == 'vit2':
        args.backbone_path = "./checkpoints/ViT-B_16.npz"
    elif args.backbone == 'mae':
        args.backbone_path = "./checkpoints/mae_pretrain_vit_base.pth"
    elif args.backbone == 'simmim':
        args.backbone_path = "./checkpoints/simmim_pretrain__vit_base__img224__800ep.pth"
    backbone = get_backbone(args)
    backbone.eval()

    extract(train_loader, test_loader, backbone, args)

def get_backbone(args):
    if args.backbone == 'moco_v1' or args.backbone == 'moco_v2':
        if args.dataset == 'cifar10':
            backbone = backbones.__dict__[args.arch](num_classes=10)
        elif args.dataset == 'cifar100':
            backbone = backbones.__dict__[args.arch](num_classes=100)

        # freeze all layers but the last fc
        for name, param in backbone.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
        # init the fc layer
        backbone.fc.weight.data.normal_(mean=0.0, std=0.01)
        backbone.fc.bias.data.zero_()
        # load from pre-trained, before DistributedDataParallel constructor
        if args.backbone_path:
            if os.path.isfile(args.backbone_path):
                print("=> loading checkpoint '{}'".format(args.backbone_path))
                checkpoint = torch.load(args.backbone_path, map_location="cpu")
                # rename moco pre-trained keys
                state_dict = checkpoint['state_dict']
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                        # remove prefix
                        state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]
                msg = backbone.load_state_dict(state_dict, strict=False)
                assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
                print("=> loaded pre-trained model '{}'".format(args.backbone_path))
            else:
                print("=> no checkpoint found at '{}'".format(args.backbone_path))

    elif args.backbone == 'clip':
        assert args.arch == 'resnet50'
        backbone, _ = clip.load('RN50', device=args.device, download_root=args.backbone_path)
        for name, param in backbone.named_parameters():
            param.requires_grad = False
    
    elif args.backbone == 'simclr_v1':
        assert args.arch == 'resnet50'
        backbone = resnet50x1()
        checkpoint = torch.load(args.backbone_path, map_location='cpu')
        backbone.load_state_dict(checkpoint['state_dict'])
        for name, param in backbone.named_parameters():
            param.requires_grad = False

    elif args.backbone == 'simclr_v2':
        assert args.arch == 'resnet50'
        backbone, _ = get_resnet(*name_to_params(args.backbone_path))
        backbone.load_state_dict(torch.load(args.backbone_path)['resnet'])
        for name, param in backbone.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
    
    elif args.backbone == 'resnet':
        assert args.arch == 'resnet50'
        backbone = resnet50(pretrained=True, progress=True)
        for name, param in backbone.named_parameters():
                param.requires_grad = False
    
    elif args.backbone == 'vit' or args.backbone == 'vit2':
        backbone = timm.create_model('vit_base_patch16_224.augreg_in21k', checkpoint_path=args.backbone_path)
        for name, param in backbone.named_parameters():
            if name not in ['head.weight', 'head.bias']:
                param.requires_grad = False

    elif args.backbone == 'mae':
        if args.dataset == 'cifar10':
            backbone = vit_base_patch16(num_classes=10,global_pool=False)
        elif args.dataset == 'cifar100':
            backbone = vit_base_patch16(num_classes=100,global_pool=False)

        checkpoint = torch.load(args.backbone_path, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.backbone_path)
        checkpoint_model = checkpoint['model']
        state_dict = backbone.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(backbone, checkpoint_model)

        # load pre-trained model
        msg = backbone.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        for name, param in backbone.named_parameters():
            param.requires_grad = False

    elif args.backbone == 'simmim':
        if args.dataset == 'cifar10':
            backbone = build_vit_B_16(num_classes=10)
        elif args.dataset == 'cifar100':
            backbone = build_vit_B_16(num_classes=100)
        load_pretrained(backbone, args)
        for name, param in backbone.named_parameters():
            param.requires_grad = False
    
    else:
        print('fault')

    return backbone.to(args.device)

def extract(train_loader, test_loader, backbone, args):
    train_features = []
    train_labels = []
    val_features = []
    val_labels = []
    with torch.no_grad():
        for (x, target) in tqdm(train_loader, total= len(train_loader)):
            x = x.to(args.device)

            if args.backbone == 'moco_v1' or args.backbone == 'moco_v2' or args.backbone == 'resnet' or args.backbone == 'simclr_v2' or args.backbone == 'simclr_v1':
                x = backbone(x)[0] if type(backbone(x)) is tuple else backbone(x)
            elif args.backbone == 'clip':
                x = backbone.encode_image(x)
            elif args.backbone == 'vit' or args.backbone == 'vit2':
                x = backbone.forward_features(x)[:,0]
            elif args.backbone in ['mae', 'simmim']:
                x = backbone.forward_features(x)
            train_features.append(x.detach().cpu().numpy())
            train_labels.append(target.numpy())
        train_features, train_labels = np.concatenate(train_features), np.concatenate(train_labels)
        train_features = np.concatenate([train_features, train_labels.reshape((-1,1))], axis=1)
        np.save(os.path.join(args.features_dir, 'train_features.npy'), train_features)
        print(train_features.shape)
        print(train_features.max(), train_features.min())

        for (x, target) in tqdm(test_loader, total= len(test_loader)):
            x = x.to(args.device)
            if args.backbone == 'moco_v1' or args.backbone == 'moco_v2' or args.backbone == 'resnet' or args.backbone == 'simclr_v2' or args.backbone == 'simclr_v1':
                x = backbone(x)[0] if type(backbone(x)) is tuple else backbone(x)
            elif args.backbone == 'clip':
                x = backbone.encode_image(x)
            elif args.backbone == 'vit' or args.backbone == 'vit2':
                x = backbone.forward_features(x)[:,0]
            elif args.backbone in ['mae', 'simmim']:
                x = backbone.forward_features(x)

            val_features.append(x.detach().cpu().numpy())
            val_labels.append(target.numpy())

        val_features, val_labels = np.concatenate(val_features), np.concatenate(val_labels)
        val_features = np.concatenate([val_features, val_labels.reshape((-1,1))], axis=1)
        np.save(os.path.join(args.features_dir, 'val_features.npy'), val_features)
        print(val_features.shape)

    return

if __name__ == '__main__':
    main()
