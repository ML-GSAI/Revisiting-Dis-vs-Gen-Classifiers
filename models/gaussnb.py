# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from math import pi
import sys
import time

class GaussianNB():

    def __init__(self, val_epsilon=1e-9):
        self.val_epsilon = val_epsilon
        self.features = None
        self.K = None
        self.mus = None
        self.vars_groupby = None
        self.prior = None
        self.counts = None
    
    def get_vars(self, samples:pd.DataFrame):
        samples = samples.iloc[:,:-1]
        vars = np.var(samples, axis=0) + self.val_epsilon
        return vars

    def fit(self, X_train:np.ndarray, y_train:np.ndarray):
        X_train = pd.DataFrame(X_train)
        y_train = pd.DataFrame(y_train)
        if self.features is None:
            self.features = X_train.shape[1]
        if self.K is None:
            self.K = len(set(y_train.values.squeeze()))
            # print(self.K)
        
        train_set = pd.concat((X_train, y_train), axis=1, ignore_index=True)
        
        if self.vars_groupby is None and self.mus is None and self.prior is None and self.counts is None:
            grouped = train_set.groupby([train_set.shape[1]-1])
            self.mus = grouped.agg('mean').values
            self.counts = grouped.agg('count').loc[:,0].values
            self.prior = self.counts / train_set.shape[0]
            self.vars_groupby = grouped.apply(self.get_vars).values

        else:
            grouped = train_set.groupby([train_set.shape[1]-1])
            mus_new = grouped.agg('mean').values
            counts_new = grouped.agg('count').loc[:,0].values
            vars_groupby_new = grouped.apply(self.get_vars).values


            counts_total = self.counts + counts_new
            mus_total = (self.counts.reshape((-1,1)) * self.mus + counts_new.reshape((-1,1)) * mus_new) / counts_total.reshape((-1,1))
            
            old_ssd = self.counts.reshape((-1,1)) * self.vars_groupby
            new_ssd = counts_new.reshape((-1,1)) * vars_groupby_new
            total_ssd = old_ssd + new_ssd + (counts_new * self.counts / counts_total).reshape((-1,1)) * (self.mus - mus_new) ** 2
            total_var_groupby = total_ssd / counts_total.reshape((-1,1))

            self.mus = mus_total
            self.vars_groupby = total_var_groupby
            self.counts = counts_total
            self.prior = self.counts / np.sum(self.counts)


    def predict_gaussian_likehood(self, X_test: np.ndarray):
        vars = np.sum(self.vars_groupby * self.prior.reshape(-1,1), axis=0)
        joint_log_likelihood = []
        # n_ij = 0
        for i in range(self.K):
            n_ij = -0.5 * np.sum(((X_test - self.mus[i, :]) ** 2) / (vars.reshape(1,-1)) + np.log(2*pi*vars.reshape(1,-1)), 1) 
            joint_log_likelihood.append(n_ij)
        joint_log_likelihood = np.array(joint_log_likelihood).T
        # print(joint_log_likelihood)
        return joint_log_likelihood
    
    def predict(self, X_test: np.ndarray):
        vars = np.sum(self.vars_groupby * self.prior.reshape(-1,1), axis=0)
        joint_log_likelihood = []
        # n_ij = 0
        for i in range(self.K):
            jointi = np.log(self.prior[i])
            n_ij = -0.5 * np.sum(((X_test - self.mus[i, :]) ** 2) / (vars.reshape(1,-1)), 1)
            joint_log_likelihood.append(jointi + n_ij)
        joint_log_likelihood = np.array(joint_log_likelihood).T
        # print(joint_log_likelihood)
        return np.argmax(joint_log_likelihood, axis=1)
        
    def score(self, X_test, y_test):
        preds = self.predict(X_test)
        # print(preds)
        score = np.sum(preds == y_test) / len(y_test)
        return score


def main():
    from sklearn.datasets import make_blobs, make_classification
    from sklearn.model_selection import train_test_split
    features = 5
    X, y = make_blobs(n_samples=100000, n_features=features, centers=[features*[-1], features*[1]],
                  cluster_std=[np.sqrt(features), 0.01*np.sqrt(features)], random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=90000)

    X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_train, y_train, train_size=0.5)

    t1 = time.time()
    nb = GaussianNB()

    nb.fit(X_train_1, y_train_1)
    print(nb.vars_groupby)
    nb.fit(X_train_2, y_train_2)
    print(nb.vars_groupby)
    print(nb.score(X_test, y_test))

if __name__ == '__main__':
    main()