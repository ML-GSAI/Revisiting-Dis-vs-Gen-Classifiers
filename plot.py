import argparse
from utils.vis import *

parser = argparse.ArgumentParser(description='plot')
parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10','cifar100'],
                    help='dataset the features belong to.')
parser.add_argument('--backbone', default='clip', type=str, choices=['moco_v1', 'moco_v2', 'clip','resnet', 'vit', 'simclr_v2', 'simclr_v1', 'vit2', 'mae', 'simmim'],
                    help='pretrained backbone.')
parser.add_argument('--lr_path', default='', type=str,
                    help='dataset the features belong to.')
parser.add_argument('--nb_diag_path', default='', type=str,
                    help='pretrained backbone.')
parser.add_argument('--nb_full_path', default=None, type=str,
                    help='model.')
parser.add_argument('--nb_low_rank_path', default=None, type=str,
                    help='model.')
parser.add_argument('--pic_dir', default='', type=str,
                    help='model.')
parser.add_argument('--mode', default='mean_var', type=str, 
                    choices=['long','short', 'sigmas', 'kl', 'var_likelihood_diff'],
                    help='mode.')


if __name__ == '__main__':
    args = parser.parse_args()
    if args.mode == 'sigmas':
        stats_features_sigmas(args.dataset, args.backbone)
    elif args.mode == 'kl':
        stats_features_KL(args.dataset, args.backbone)
    elif args.mode == 'var_likelihood_diff':
        stats_features_var_likelihood_diff(args.dataset, args.backbone)
    elif args.mode in ['long', 'short']:
        show_deep_results(args.dataset, args.backbone, args.lr_path, args.nb_diag_path, args.nb_full_path, args.nb_low_rank_path, args.pic_dir, args.mode)