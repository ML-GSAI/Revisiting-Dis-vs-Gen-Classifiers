import argparse
import numpy as np
from tqdm import tqdm
import os
from math import ceil

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from data.dataset import SimDataset
from models.logistic import MulticlassLogisticRegressionModel, LogisticRegressionModel
from sklearn.linear_model import LogisticRegression
from models.gaussnb import GaussianDA, GaussianNB, GaussianNB_puls_low_rank

from utils.tools import *
from utils.vis import save_vis_sim

parser = argparse.ArgumentParser(description='Simulation')
parser.add_argument('--data_root', default='./datasets/sim_normal', type=str,
                    help='data dir.')
parser.add_argument('--K', default=2, type=int, metavar='N',
                    help='number of class')
parser.add_argument('--n', default=100, type=int, metavar='N',
                    help='feature dimension')
parser.add_argument('--t', default=1, type=int, metavar='N',
                    help='id')
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
parser.add_argument('--lr', default=1., type=float,
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
parser.add_argument('--mode', default=None, type=str,
                    help='easy or hard.')

bayes_error = {}
bayes_error[2] = {
    2: 0.06655,
    4: 0.02924,
    10: 0.00483,
    20: 0.00037,
    40: 0,
    100: 0,
    200: 0,
    400: 0,
    1000: 0
}
bayes_error[3] = bayes_error[5]= bayes_error[7] {
    40: 0,
    100: 0,
    200: 0,
    400: 0,
    1000: 0,
    2000: 0,
    4000: 0,
    10000: 0
}


def main():
    args = parser.parse_args()
    data_dir = os.path.join(args.data_root, 'K' + str(args.K) + '_n' + str(args.n))
    log_dir = os.path.join('./log', args.mode, str(args.K), str(args.n), args.model)
    if args.model == 'lr_bgfs':
        log_dir = os.path.join(log_dir, 'C' + str(args.C))
    elif args.model == 'lr_sgd':
        log_dir = os.path.join(log_dir, 'lr' + str(args.lr) + '_wd' + str(args.wd))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    loss_path = os.path.join(log_dir, 'loss_%s.npy' % (args.t))
    pic_path = os.path.join(log_dir, 'vis_%s.png' % (args.t))

    logger = get_console_file_logger(name='offline, %s, %s, %s' % (args.K, args.n, args.model), logdir=log_dir)
    logger.info(args._get_kwargs())

    m_step = list(range(2 * args.K, 100 * args.n, ceil(np.log(args.n))))

    train_set = SimDataset(root=data_dir, features=args.n, K=args.K)
    test_set = SimDataset(root=data_dir, features=args.n, K=args.K, train=False)
    test_loader = DataLoader(test_set, batch_size=args.bs, shuffle=False)

    if args.model == 'lr_sgd':
        loss_func = nn.CrossEntropyLoss()
        args.device = torch.device('cuda', args.gpu) if args.gpu is not None else 'cuda'
    
    errors = np.zeros((args.repeat, len(m_step)))

    for m_idx, m in enumerate(m_step):
        if args.model == 'lr_sgd':
            errors = train_fix_m_sgd(train_set, test_loader, loss_func, m_idx, m, errors, logger, args)
        else:
            errors = train_fix_m_no_sgd(train_set, test_loader, m_idx, m, errors, logger, args)
        logger.info('m = %d, m_idx = %d' % (m, m_idx))
        logger.info(errors[:,m_idx])
        np.save(loss_path, errors)
        if np.mean(errors[:,m_idx]) < bayes_error[args.K][args.n] + 0.01:
            break
    
    save_vis_sim(m_step, errors, pic_path, args)

def get_model(args):
    if args.model == 'lr_bgfs':
        model = LogisticRegression(penalty='l2', C=args.C, solver='lbfgs', max_iter=1000)
    elif args.model == 'lr_sgd':
        if args.K == 2:
            model = LogisticRegressionModel(args.n)
        else:
            model = MulticlassLogisticRegressionModel(args.n, args.K)
    elif args.model == 'nb_diag':
        model = GaussianNB(val_epsilon=args.epsilon)
    else:
        print('fault')
    return model
    
def train_fix_m_no_sgd(train_set, test_loader, m_idx, m, errors, logger, args):
    i = 0
    flag = False
    while flag == False:
        for _ in tqdm(range(10)):
            train_set_m, _ = random_split(train_set, [m, len(train_set)-m])
            if args.model == 'lr_bgfs':
                train_loader = DataLoader(train_set_m, batch_size=m, shuffle=True)
            else:
                train_loader = DataLoader(train_set_m, batch_size=args.bs, shuffle=True)

            for _, label in train_loader:
                label = label.numpy()
                break
            if len(set(list(label))) < args.K:
                continue
            
            i += 1
            model = get_model(args)

            for x, label in train_loader:
                x = x.numpy()
                label = label.numpy()
                model.fit(x, label)

            acc = 0
            with torch.no_grad():
                for x, label in test_loader:
                    x = x.numpy()
                    label = label.numpy()
                    preds = model.predict(x)
                    acc += (preds == label).sum()

                error = 1 - acc / 10000
            errors[i-1, m_idx] = error
            if i > args.repeat - 1:
                flag = True
                break
    return errors

def train_fix_m_sgd(train_set, test_loader, loss_func, m_idx, m, errors, logger, args):
    i = 0
    flag = False
    while flag == False:
        for _ in tqdm(range(10)):
            train_set_m, _ = random_split(train_set, [m, len(train_set)-m])
            train_loader = DataLoader(train_set_m, batch_size=args.bs, shuffle=True)
            for _, label in train_loader:
                label = label.numpy()
                break
            if len(set(list(label))) < args.K:
                continue
            i += 1

            best_error_test = 1
            early_stop = 0

            model = get_model(args)
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
                    with torch.no_grad():
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
                        if early_stop > 14:
                            break                    
                # lr_scheduler.step()
            errors[i-1, m_idx] = best_error_test
            if i > args.repeat - 1:
                flag = True
                break
    return errors

if __name__ == '__main__':
    main()