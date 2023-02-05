import argparse
import os
import numpy as np

from sklearn.linear_model import LogisticRegression
from models.logistic import MulticlassLogisticRegressionModel
from models.gaussnb import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm

from utils.tools import *
from utils.vis import save_vis


parser = argparse.ArgumentParser(description='Offline Linear eval')
parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10','cifar100'],
                    help='dataset the features belong to.')
parser.add_argument('--backbone', default='clip', type=str, choices=['moco_v1', 'moco_v2', 'clip','resnet','vit', 'simclr_v2', 'simclr_v1', 'vit2', 'mae', 'simmim'],
                    help='pretrained backbone.')
parser.add_argument('--model', default='lr_bgfs', type=str, choices=['lr_bgfs', 'lr_sgd', 'nb_diag'],
                    help='model.')
parser.add_argument('--C', default=1, type=float,
                    help='peanlty of l2, lr_bgfs.')
parser.add_argument('--epsilon', default=1e-9, type=float,
                    help='var smoothing of naive Bayes.')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--bs', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', default=30., type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='wd')
parser.add_argument('--repeat', default=5, type=int, metavar='N',
                    help='repeat times')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--minmax', dest='minmax', action='store_true',
                    help='scaler to [0,1]')

def main():
    args = parser.parse_args()
    data_dir = os.path.join('./datasets', args.backbone, args.dataset)
    if not args.minmax:
        log_dir = os.path.join('./log', 'offline', args.backbone, args.dataset, args.model)
    else:
        log_dir = os.path.join('./log', 'offline', args.backbone + '_minmax', args.dataset, args.model)
    if args.model == 'lr_bgfs':
        log_dir = os.path.join(log_dir, 'C' + str(args.C))
    elif args.model == 'lr_sgd':
        log_dir = os.path.join(log_dir, 'lr' + str(args.lr) + '_wd' + str(args.wd))

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    loss_path = os.path.join(log_dir, 'loss.npy')
    pic_path = os.path.join(log_dir, 'vis.png')

    logger = get_console_file_logger(name='offline, %s, %s, %s' % (args.backbone, args.dataset, args.model), logdir=log_dir)
    logger.info(args._get_kwargs())
    if args.dataset == 'cifar10':
        K = 10
        m_step = [20,50,100,200,500,1000,2000,5000,10000,20000,30000,50000]
    else:
        K = 100
        m_step = [3*K,5*K,10*K,20*K,50*K,100*K,200*K,500*K]

    train_set_path = os.path.join(data_dir, 'train_features.npy')
    test_set_path = os.path.join(data_dir, 'val_features.npy')
    train_set = np.load(train_set_path)
    test_set = np.load(test_set_path)
    X_train, y_train = train_set[:,0:-1], train_set[:,-1]
    X_test, y_test = test_set[:,0:-1], test_set[:,-1]

    
    if args.model == 'lr_sgd':
        loss_func = nn.CrossEntropyLoss()
        args.device = torch.device('cuda', args.gpu) if args.gpu is not None else 'cuda'
        X_test, y_test = torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long()
        test_set = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_set, batch_size=args.bs, shuffle=False)

    errors = np.zeros((args.repeat, len(m_step)))

    for m_idx, m in enumerate(m_step):
        if args.model == 'lr_sgd':
            errors = train_fix_m_sgd(X_train, y_train, test_loader, loss_func, m_idx, m, K, errors, logger, args)
        else:
            errors = train_fix_m_no_sgd(X_train, y_train, X_test, y_test, m_idx, m, K, errors, logger, args)
        logger.info('m = '+ str(m))
        logger.info(errors)
        np.save(loss_path, errors)
    
    save_vis(m_step, errors, pic_path, args)

def get_model(args):
    if args.model == 'lr_bgfs':
        model = LogisticRegression(penalty='l2', C=args.C, solver='lbfgs', max_iter=1000)
    elif args.model == 'lr_sgd':
        if args.backbone in ['moco_v1', 'moco_v2', 'simclr_v1', 'simclr_v2', 'resnet']:
            if args.dataset == 'cifar10':
                model = MulticlassLogisticRegressionModel(features=2048, K=10)
            elif args.dataset == 'cifar100':
                model = MulticlassLogisticRegressionModel(features=2048, K=100)
        elif args.backbone == 'clip' and args.dataset == 'cifar10':
            model = MulticlassLogisticRegressionModel(features=1024, K=10)
        elif args.backbone == 'clip' and args.dataset == 'cifar100':
            model = MulticlassLogisticRegressionModel(features=1024, K=100)
        elif args.backbone in ['vit', 'vit2', 'mae', 'simmim']:
            if args.dataset == 'cifar10':
                model = MulticlassLogisticRegressionModel(features=768, K=10)
            elif args.dataset == 'cifar100':
                model = MulticlassLogisticRegressionModel(features=768, K=100)
    elif args.model == 'nb_diag':
        model = GaussianNB(val_epsilon=args.epsilon)
    else:
        print('fault')
    return model
    
def train_fix_m_no_sgd(X_train, y_train, X_test, y_test, m_idx, m, K, errors, logger, args):
    i = 0
    flag = False
    while flag == False:
        for _ in tqdm(range(10)):
            if m < 50000:
                X_train_m, _, y_train_m, _ = train_test_split(X_train, y_train, train_size=m)
            else:
                X_train_m, y_train_m = X_train, y_train
            if len(set(list(y_train_m))) < K:
                continue
            if args.minmax:
                scaler = MinMaxScaler()
                X_train_m = scaler.fit_transform(X_train_m)
                X_test_temp = scaler.transform(X_test)
                
            i += 1
            model = get_model(args)
            model.fit(X_train_m, y_train_m)
            errors[i-1, m_idx] = (1 - model.score(X_test_temp, y_test))
            if i > args.repeat - 1:
                flag = True
                break
    return errors

def train_fix_m_sgd(X_train, y_train, test_loader, loss_func, m_idx, m, K, errors, logger, args):
    i = 0
    flag = False
    while flag == False:
        for _ in tqdm(range(10)):
            if m < 50000:
                X_train_m, _, y_train_m, _ = train_test_split(X_train, y_train, train_size=m)
            else:
                X_train_m, y_train_m = X_train, y_train
            if len(set(list(y_train_m))) < K:
                continue
            i += 1
            X_train_m, y_train_m = torch.from_numpy(X_train_m).float(), torch.from_numpy(y_train_m).long()
            train_set = TensorDataset(X_train_m, y_train_m)
            train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True)
                    
            best_error_test = 1
            early_stop = 0

            model = get_model(args).to(args.device)
            model.apply(model.init_weights)
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
            # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1,gamma=0.9)
                    
            for epoch in tqdm(range(args.epochs)):
                model.train()
                adjust_learning_rate(optimizer, epoch, args)
                for x, label in train_loader:
                    x = x.to(args.device)
                    label = label.to(args.device)
                    pred = model(x).squeeze()
                    loss = loss_func(pred, label)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if (epoch + 1) % 5 == 0:
                    correct = 0
                    model.eval()
                    for x, label in test_loader:
                        x = x.to(args.device)
                        label = label.to(args.device)
                        pred = model(x).argmax(axis=1).squeeze()
                        correct += (pred == label).sum().item()
                    error_test = 1 - correct / 10000
                    logger.info('epoch = %d, test_error = %.6f' % (epoch+1, error_test))

                    if error_test < best_error_test:
                        early_stop = 0
                        best_error_test = error_test
                    else:
                        early_stop += 1
                        if early_stop > 100:
                            break
                    
                # lr_scheduler.step()
            errors[i-1, m_idx] = best_error_test
            if i > args.repeat - 1:
                flag = True
                break
    return errors

if __name__ == '__main__':
    main()