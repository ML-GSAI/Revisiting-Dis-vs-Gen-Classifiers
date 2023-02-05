import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import adjusted_mutual_info_score
import futureproof
import sys
from models.gaussnb import GaussianNB

def save_vis_sim(m_step, errors, pic_path, args):
    plt.plot(m_step, np.mean(errors, axis=0), c='r', linewidth=1, label=args.model)
    plt.title(str(args.K) + ', ' + str(args.n))
    plt.xlabel('m')
    plt.ylabel('error')
    plt.legend()
    plt.savefig(pic_path)
    plt.close()

def save_vis(m_step, errors, pic_path, args):
    plt.plot(m_step, np.mean(errors, axis=0), c='r', linewidth=1, label=args.model)
    plt.title(args.backbone + ', ' + args.dataset)
    plt.xlabel('m')
    plt.ylabel('error')
    plt.legend()
    plt.savefig(pic_path)
    plt.close()

def show_deep_results(dataset, backbone, lr_path, nb_diag_path, nb_full_path, nb_low_rank_path, pic_dir, mode):
    if dataset == 'cifar10':
        if mode == 'long':
            m_step = [20,50,100,200,500,1000,2000,5000,10000,20000,30000,50000]
        else:
            m_step = [20,50,100,200,500,1000]
    else:
        K = 100
        if mode == 'long':
            m_step = [3*K,5*K,10*K,20*K,50*K,100*K,200*K,500*K]
        else:
            m_step = [3*K,5*K,10*K,20*K]

    error_lr = np.load(lr_path)
    print(error_lr.shape)
    error_nb_diag = np.load(nb_diag_path)
    if nb_full_path is not None:
        error_nb_full = np.load(nb_full_path)
    if nb_low_rank_path is not None:
        error_nb_low_rank = np.load(nb_low_rank_path)

    plt.figure()
    plt.plot(m_step, np.mean(error_lr[:,:len(m_step)], axis=0), marker='s', color='g', label='LR')
    plt.plot(m_step, np.mean(error_nb_diag[:,:len(m_step)], axis=0), marker='o', color='r', label='NB')
    if nb_full_path is not None:
        plt.plot(m_step, np.mean(error_nb_full[:,:len(m_step)], axis=0), linewidth=1, label='NB_full')
    if nb_low_rank_path is not None:
        plt.plot(m_step, np.mean(error_nb_low_rank[:,:len(m_step)], axis=0), linewidth=1, label='NB_diag_low rank')
    # plt.title(backbone + ', ' + dataset)
    plt.rcParams['font.family'] = ['Times New Roman']
    plt.legend()
    plt.xlabel('m', labelpad=8, fontsize = 13)
    plt.ylabel('error', labelpad=8, fontsize = 13)
    plt.yticks(fontproperties = 'Times New Roman', size = 13)
    plt.xticks(fontproperties = 'Times New Roman', size = 13)
    pic_path = os.path.join(pic_dir, '%s.png' % (mode))
    plt.savefig(pic_path,bbox_inches='tight', dpi=800)
    plt.close()


def get_vars(samples:pd.DataFrame):
    samples = samples.iloc[:,:-1]
    vars = np.var(samples, axis=0) + 1e-9
    return vars

def stats_features_sigmas(dataset, backbone):
    data_dir = os.path.join('./datasets', backbone, dataset)
    log_dir = os.path.join('./stats', 'sigmas_minmax', backbone, dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    vars_path = os.path.join(log_dir, 'vars.npy')
    vars_pic_path = os.path.join(log_dir, 'vars.png')
    vars_p_path = os.path.join(log_dir, 'vars.csv')

    train_set_path = os.path.join(data_dir, 'train_features.npy')
    train_features = np.load(train_set_path)
    print(train_features.shape)
    X_train, y_train = train_features[:,0:-1], train_features[:,-1]
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    train_features = np.concatenate([X_train, y_train.reshape(-1,1)], axis=1)

    train_features = pd.DataFrame(train_features)
    grouped = train_features.groupby([train_features.shape[1]-1])

    counts = grouped.agg('count').loc[:,0].values
    prior = counts / train_features.shape[0]
    vars_groupby = grouped.apply(get_vars).values
    vars = np.sum(vars_groupby * prior.reshape(-1,1), axis=0)

    np.save(vars_path, vars)

    vars_p_value = np.zeros((2, 11))
    for p_idx, p in enumerate(range(0,101,10)):
        vars_p_value[0][p_idx] = p
        vars_p_value[1][p_idx] = np.percentile(vars, p)
    vars_p_value = pd.DataFrame(vars_p_value)
    vars_p_value.to_csv(vars_p_path, index=False)

    plt.figure()
    plt.rcParams['font.family'] = ['Times New Roman']
    plt.xlabel(r'$\sigma_i^2$', labelpad=8, fontsize = 13)
    plt.ylabel('counting', labelpad=8, fontsize = 13)
    # 设置字体
    plt.yticks(fontproperties = 'Times New Roman', size = 13)
    plt.xticks(fontproperties = 'Times New Roman', size = 13)

    plt.hist(vars, bins=100)
    plt.savefig(vars_pic_path, bbox_inches='tight', dpi=800)
    plt.close()

from math import pi

def stats_features_KL(dataset, backbone):
    data_dir = os.path.join('./datasets', backbone, dataset)
    log_dir = os.path.join('./stats', 'KL_gaussian', backbone, dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    kl_diff_path = os.path.join(log_dir, 'kl_diff.npy')
    kl_pic_path = os.path.join(log_dir, 'kl.png')
    kl_p_path = os.path.join(log_dir, 'kl.csv')

    train_set_path = os.path.join(data_dir, 'train_features.npy')
    train_features = np.load(train_set_path)
    print(train_features.shape)
    X_train, y_train = train_features[:,0:-1], train_features[:,-1]
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    joint_log_likelihood = nb.predict_gaussian_likehood(X_train)
    
    train_features = np.concatenate([joint_log_likelihood, y_train.reshape(-1,1)], axis=1)
    train_features = train_features[train_features[:,-1].argsort()]
    train_features_grouped = np.split(train_features[:,:-1], 10, axis=0)
    
    n = train_features.shape[1] - 1
    K = len(train_features_grouped)

    kl_diff = []
    for k in range(10):
        for k1 in range(10):
            for k2 in range(10):
                if k1 != k2:
                    kl_diff.append(np.mean(train_features_grouped[k][:,k1] - train_features_grouped[k][:,k2]))
    kl_diff = np.array(kl_diff)
    kl_diff = np.abs(kl_diff) / n
    np.save(kl_diff_path, kl_diff)

    kl_p_value = np.zeros((2, 11))
    for p_idx, p in enumerate(range(0,101,10)):
        kl_p_value[0][p_idx] = p
        kl_p_value[1][p_idx] = np.percentile(kl_diff, p)
    kl_p_value = pd.DataFrame(kl_p_value)
    kl_p_value.to_csv(kl_p_path, index=False)

    plt.figure()
    plt.rcParams['font.family'] = ['Times New Roman']
    plt.xlabel(r'$\vert\beta_{k_1,k_2,k}\vert$', labelpad=8, fontsize = 13)
    plt.ylabel('counting', labelpad=8, fontsize = 13)
    # 设置字体
    plt.yticks(fontproperties = 'Times New Roman', size = 13)
    plt.xticks(fontproperties = 'Times New Roman', size = 13)
    plt.xlabel(r'$\vert\beta_{k_1,k_2,k}\vert$')
    plt.ylabel('counting')

    plt.hist(kl_diff, bins=100)
    plt.savefig(kl_pic_path, bbox_inches='tight', dpi=800)
    plt.close()


def stats_features_var_likelihood_diff(dataset, backbone):
    data_dir = os.path.join('./datasets', backbone, dataset)
    log_dir = os.path.join('./stats', 'var_likelihood_diff', backbone, dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    kl_diff_path = os.path.join(log_dir, 'var_likelihood_diff.npy')
    kl_pic_path = os.path.join(log_dir, 'var_likelihood_diff.png')
    kl_p_path = os.path.join(log_dir, 'var_likelihood_diff.csv')

    train_set_path = os.path.join(data_dir, 'train_features.npy')
    train_features = np.load(train_set_path)
    print(train_features.shape)
    X_train, y_train = train_features[:,0:-1], train_features[:,-1]
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    joint_log_likelihood = nb.predict_gaussian_likehood(X_train)
    
    train_features = np.concatenate([joint_log_likelihood, y_train.reshape(-1,1)], axis=1)
    train_features = train_features[train_features[:,-1].argsort()]
    train_features_grouped = np.split(train_features[:,:-1], 10, axis=0)
    
    n = train_features.shape[1] - 1
    K = len(train_features_grouped)

    kl_diff = []
    for k in range(10):
        for k1 in range(10):
            for k2 in range(10):
                if k1 != k2:
                    kl_diff.append(np.var(train_features_grouped[k][:,k1] - train_features_grouped[k][:,k2]))
    kl_diff = np.array(kl_diff)
    kl_diff = np.abs(kl_diff) / n
    np.save(kl_diff_path, kl_diff)

    kl_p_value = np.zeros((2, 11))
    for p_idx, p in enumerate(range(0,101,10)):
        kl_p_value[0][p_idx] = p
        kl_p_value[1][p_idx] = np.percentile(kl_diff, p)
    kl_p_value = pd.DataFrame(kl_p_value)
    kl_p_value.to_csv(kl_p_path, index=False)

    plt.figure()
    plt.rcParams['font.family'] = ['Times New Roman']
    plt.xlabel(r'$\vert\alpha_{k_1,k_2,k}\vert$', labelpad=8, fontsize = 13)
    plt.ylabel('counting', labelpad=8, fontsize = 13)
    
    plt.yticks(fontproperties = 'Times New Roman', size = 13)
    plt.xticks(fontproperties = 'Times New Roman', size = 13)

    plt.hist(kl_diff, bins=100)
    plt.savefig(kl_pic_path, bbox_inches='tight', dpi=800)
    plt.close()


if __name__ == '__main__':
    pass