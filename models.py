#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy as sp

from utils import read_data,evaluate, find_optimal_C

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from gensim.models import Word2Vec

from utils import compute_M

import time

def train_rle(A,T,U,d,lamb,verbose=True):
    if verbose:
        print('RLE - d=%d' % d)
        
    start = time.time()
    
    G = sp.sparse.csr_matrix(normalize(compute_M(A), norm='l1', axis=1), dtype=np.float32)
    T = sp.sparse.csr_matrix(normalize(T, norm='l1', axis=1), dtype=np.float32)
    
    Tprime = G @ T

    O = T * (1 - lamb) + Tprime * lamb

    rle_embeddings = O @ U

    rle_embeddings = normalize(rle_embeddings,axis=0)

    training_time = time.time() - start
    if verbose:
        print('Training time: %.1f' % training_time)
    
    return rle_embeddings, training_time


def train_tadw(A, X, d=160, order=2, iter_max=20, alpha=0.2, lamb=0.2, verbose=True):
    """
    TADW (Text-Associated DeepWalk) implementation
    
    Args:
        A: Adjacency matrix (sparse)
        X: Text feature matrix (TF-IDF)
        d: embedding dimension
        order: order of adjacency matrix powers to use
        iter_max: maximum iterations for optimization
        alpha: learning rate
        lamb: regularization parameter
        verbose: print training info
    
    Returns:
        W: node embeddings
        training_time: time taken for training
    """
    if verbose:
        print('TADW - d=%d, order=%d' % (d, order))
    
    start = time.time()
    
    # Convert to dense if needed and ensure proper format
    if sp.sparse.issparse(A):
        A = A.toarray()
    if sp.sparse.issparse(X):
        X = X.toarray()
    
    A = A.astype(np.float32)
    X = X.astype(np.float32)
    
    n, ft_size = X.shape
    
    # Compute matrix M = sum of powers of adjacency matrix
    # M = A^1 + A^2 + ... + A^order
    M = np.zeros_like(A)
    A_power = A.copy()
    
    for i in range(1, order + 1):
        if i > 1:
            A_power = A_power @ A
        M += A_power
    
    # Add identity to include self-loops
    M += np.eye(n)
    
    # Initialize embeddings randomly
    W = np.random.normal(0, 0.1, (n, d)).astype(np.float32)
    H = np.random.normal(0, 0.1, (ft_size, d)).astype(np.float32)
    
    # Optimization using alternating least squares
    for iteration in range(iter_max):
        # Update W
        XH = X @ H
        
        # For each node, solve least squares problem
        for i in range(n):
            # Get neighbors with weights
            neighbors = np.where(M[i] > 0)[0]
            if len(neighbors) == 0:
                continue
                
            # Build system for node i
            # Minimize ||M_i - W_i @ (W_neighbors + XH_i)||^2 + lambda ||W_i||^2
            
            # Construct target vector
            y_i = M[i, neighbors]
            
            # Construct feature matrix (neighbors + text features)
            if len(neighbors) > 0:
                X_i = W[neighbors] + XH[i].reshape(1, -1)
                
                # Regularized least squares
                try:
                    A_reg = X_i.T @ X_i + lamb * np.eye(d)
                    b_reg = X_i.T @ y_i
                    W[i] = np.linalg.solve(A_reg, b_reg)
                except np.linalg.LinAlgError:
                    # Fallback to pseudo-inverse if singular
                    A_reg = X_i.T @ X_i + lamb * np.eye(d)
                    W[i] = np.linalg.pinv(A_reg) @ (X_i.T @ y_i)
        
        # Update H  
        for j in range(ft_size):
            # Find nodes that have this feature
            nodes_with_feature = np.where(X[:, j] > 0)[0]
            if len(nodes_with_feature) == 0:
                continue
                
            # Build system for feature j
            y_j = []
            X_j = []
            
            for i in nodes_with_feature:
                neighbors = np.where(M[i] > 0)[0]
                if len(neighbors) > 0:
                    for neighbor in neighbors:
                        y_j.append(M[i, neighbor] - W[i] @ W[neighbor])
                        X_j.append(X[i, j] * W[i])
            
            if len(y_j) > 0:
                y_j = np.array(y_j)
                X_j = np.array(X_j)
                
                # Regularized least squares
                try:
                    A_reg = X_j.T @ X_j + lamb * np.eye(d)
                    b_reg = X_j.T @ y_j
                    H[j] = np.linalg.solve(A_reg, b_reg)
                except np.linalg.LinAlgError:
                    A_reg = X_j.T @ X_j + lamb * np.eye(d)
                    H[j] = np.linalg.pinv(A_reg) @ (X_j.T @ y_j)
    
    # Final embeddings combine structural and textual components
    tadw_embeddings = W + X @ H
    
    training_time = time.time() - start
    if verbose:
        print('Training time: %.1f' % training_time)
    
    return tadw_embeddings, training_time


def train_w2v(d,voc,raw,window,negative):

    w2v = Word2Vec(raw,vector_size=d,window=window,min_count=1,negative=negative,epochs=200)

    U = np.zeros((len(voc), d))
    for i in range(len(voc)):
        U[i,:] = w2v.wv[voc[i]]  
        
    return U

def prepare_data(d,tf,A,voc,raw,window=15,negative=5):
    
    U = train_w2v(d,voc,raw,window,negative)
    print("U done")
    
    N = A.shape[0]
         
    ind = np.argwhere(tf > 0)
    data_text = [[] for _ in range(N)]
    for dat in ind:
        data_text[dat[0]].append([dat[1],tf[dat[0],dat[1]]]) 
    print("text done")

    ind = np.argwhere(A > 0)
    data_graph = [[] for _ in range(N)]
    for dat in ind:
        data_graph[dat[0]].append([dat[1],A[dat[0],dat[1]],1]) 
        data_graph[dat[1]].append([dat[0],A[dat[0],dat[1]],-1]) 
    print("graph done")
    
    sigma = np.zeros((N,d))

    sig_def = np.std(U, axis=0)
    for i in range(N):
        nz = tf[i].nonzero()[1]
        if len(nz) < 2:
            sigma[i] = sig_def * sig_def
        else:
            sig = np.std(U[tf[i].nonzero()[1]], axis=0)
            sigma[i] = sig * sig
               
    T = sp.sparse.csr_matrix(normalize(tf, norm='l1', axis=1), dtype=np.float32)
    D = T @ U
    
    D
    
    return data_graph,data_text,sigma,D,U
    
def train_geld(d,data_graph,data_text,U,D_init,sigma_init,n_epoch=20,lamb=None,alpha=0.99,groups = None, test = False,verbose=True):
    
    if verbose:
        print('GELD - d=%d' % d)
    start = time.time()
    
    if lamb == None:
        lamb = np.power(range(1,n_epoch+1),-0.2)*0.1
    
    N = D_init.shape[0]
    
    D = D_init.copy()
    sigma = sigma_init.copy()
    
    for epoch in range(n_epoch):
        
        aine = np.random.choice(N, N, replace=False)
        for i in aine:
            mu_opt = np.zeros(d)
            denom = 0
            dat = data_graph[i]
            l_graph = len(dat)
            if(l_graph != 0):
                for obs in dat:

                    indicateur_p = ( obs[2] + 1 ) / 2
                    indicateur_n = 1 - indicateur_p
                    temp = ( indicateur_p * sigma[i] + indicateur_n * sigma[obs[0]] )
                    pond = obs[1] / temp

                    #print(pond)
                    mu_opt +=  alpha * (D[obs[0]] * pond)
                    denom += alpha *  pond

            dat = data_text[i]
            l_text = len(dat)
            if(l_text != 0) :
                for obs in dat:

                    mu_opt +=  (1 - alpha) * (U[obs[0]] * obs[1])/sigma[i]
                    denom += (1 - alpha) *  (obs[1]/sigma[i])

            if (l_text + l_graph)!=0:
                D[i] =  (1- lamb[epoch]) * D[i] + (lamb[epoch]) *  (mu_opt / denom)

        aine = np.random.choice(N, N, replace=False)
        for i in aine:
            sig_opt = np.zeros(d)
            denom = 0

            dat = data_graph[i]
            l_graph = len(dat)
            if(l_graph != 0):
                for obs in dat:
                    if obs[2] == 1:
                        dist = D[i] - D[obs[0]]
                        d2 = dist * dist
                        sig_opt += alpha * obs[1] * d2
                        denom += alpha *  obs[1]

            dat = data_text[i]
            l_text = len(dat)
            if(l_text != 0) :
                for obs in dat:

                    dist = D[i] - U[obs[0]]
                    d2 = dist * dist
                    sig_opt += (1-alpha) * obs[1] * d2
                    denom += (1-alpha) *  obs[1]
            if (l_text + l_graph)!=0:
                sigma[i] =  (1- lamb[epoch]) * sigma[i] + (lamb[epoch]) *  (sig_opt / denom)
    
        if test:
            D_norm = normalize(D, axis=1) 
            optimal_C = find_optimal_C(D_norm, groups)
            print(evaluate(D_norm,groups,0.5,C=optimal_C,verbose=False)[0])
            
    D_norm = normalize(D, axis=1)
    
    training_time = time.time() - start
    if verbose:
        print('Training time: %.1f' % training_time)
       
    return D,D_norm,sigma
