import numpy as np
import os
from sklearn.datasets import make_blobs
from tqdm import tqdm
import sys

def generate_gaussian_multiclass(features, K, m, data_dir='./datasets/sim_normal/', mode='easy', if_train=True, if_test=False):
    assert K >= 2
    features = int(features)
    data_dir = os.path.join(data_dir, 'K' + str(K) + '_n' + str(features))
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    if not os.path.exists(data_dir):
        for i in range(K):
            os.makedirs(os.path.join(train_dir, str(i)))
            os.makedirs(os.path.join(test_dir, str(i)))

    if mode == 'easy':
        cov=np.diag(int(features/2) * [features] + (features - int(features/2)) * [1])
    elif mode == 'hard':
        np.random.seed(0)
        A = 2*np.random.rand(features, features) - 1
        A = A.T @ A
        B = -np.diag(np.diag(A)) + np.diag(int(features/2) * [features] + (features - int(features/2)) * [1])
        cov = A + B
    
    if if_train == True:
        each_K = m // K
        cnt = 0
        for k in tqdm(range(K)):            
            X = np.random.multivariate_normal(
                mean=features*[-1 if k==0 else 2**(k-1)], 
                cov=cov, 
                size=int(each_K)
            )
            for i in range(X.shape[0]):
                path = os.path.join(train_dir, str(k), str(cnt) + '.npy')
                np.save(path, X[i, :])
                cnt += 1

    if if_test == True:
        batch_size = int(1e4 / K)
        cnt = 0
        for k in tqdm(range(K)):
            X = np.random.multivariate_normal(
                mean=features*[-1 if k==0 else 2**(k-1)], 
                cov=cov, 
                size=batch_size
            )
            for i in range(X.shape[0]):
                path = os.path.join(test_dir, str(k), str(cnt) + '.npy')
                np.save(path, X[i, :])
                cnt += 1

if __name__ == '__main__':
    for K in [2,3,5,7]:
        for features in [100,200,400,1000]:
            generate_gaussian_multiclass(features=features, K=K, m=100 * features, data_dir='./datasets/sim_normal/', mode='easy', if_train=True, if_test=True)
