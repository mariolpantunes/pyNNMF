# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import logging
import numpy as np


logger = logging.getLogger(__name__)


def rwnmf(X, k, alpha=0.1, tol_fit_improvement=1e-4, tol_fit_error=1e-4, num_iter=1000, seed=None):
    if isinstance(seed,int):
        np.random.seed(seed)
    
    # applies regularized weighted nmf to matrix X with k factors
    # ||X-UV^T||
    eps = np.finfo(float).eps
    early_stop = False

    # get observations matrix W
    #W = np.isnan(X)
    # print('W')
    # print(W)
    # X[W] = 0  # set missing entries as 0
    # print(X)
    #W = ~W
    # print('~W')
    # print(W)
    W = X > 0.0

    # initialize factor matrices
    #rnd = np.random.RandomState()
    #U = rnd.rand(X.shape[0], k)
    U = np.random.uniform(size=(X.shape[0], k))
    U = np.maximum(U, eps)

    V = np.linalg.lstsq(U, X, rcond=None)[0].T
    V = np.maximum(V, eps)

    Xr = np.inf * np.ones(X.shape)

    for i in range(num_iter):
        # update U
        U = U * np.divide(((W * X) @ V), (W * (U @ V.T) @ V + alpha * U))
        U = np.maximum(U, eps)
        # update V
        V = V * np.divide((np.transpose(W * X) @ U),
                          (np.transpose(W * (U @ V.T)) @ U + alpha * V))
        V = np.maximum(V, eps)

        # compute the resduals
        if i % 10 == 0:
            # compute error of current approximation and improvement
            Xi = U @ V.T
            fit_error = np.linalg.norm(X - Xi, 'fro')
            fit_improvement = np.linalg.norm(Xi - Xr, 'fro')

            # update reconstruction
            Xr = np.copy(Xi)

            # check if early stop criteria is met
            if fit_error < tol_fit_error or fit_improvement < tol_fit_improvement:
                error = fit_error
                early_stop = True
                break

    if not early_stop:
        Xr = U @ V.T
        error = np.linalg.norm(X - Xr, 'fro')

    return Xr, U, V, error


def nmf_mu(X, k, n=1000, l=1E-3, seed=None):
    if isinstance(seed,int):
        np.random.seed(seed)
    
    rows, columns = X.shape
    eps = np.finfo(float).eps

    # Create W and H
    #avg = np.sqrt(X.mean() / k)
    
    W = np.abs(np.random.uniform(size=(rows, k)))
    #W = avg * np.maximum(W, eps)
    W = np.maximum(W, eps)
    W = np.divide(W, k*W.max())

    H = np.abs(np.random.uniform(size=(k, columns)))
    #H = avg * np.maximum(H, eps)
    H = np.maximum(H, eps)
    H = np.divide(H, k*H.max())

    # Create a Mask
    M = X > 0.0

    for _ in range(n):
        W = np.multiply(W, np.divide((M*X)@H.T-l*np.linalg.norm(W, 'fro'), (M*(W@H))@H.T))
        W = np.maximum(W, eps)
        H = np.multiply(H, np.divide(W.T@(M*X)-l*np.linalg.norm(H, 'fro'), W.T@(M*(W@H))))
        H = np.maximum(H, eps)

        Xr = W @ H
        cost = np.linalg.norm((M*X) - (M*Xr), 'fro')
        if cost <= l:
            break
    
    return Xr, W, H, cost


def nmf_mu_kl(X, k, n=100, l=1E-3, seed=None):
    if isinstance(seed,int):
        np.random.seed(seed)
    
    rows, columns = X.shape
    eps = np.finfo(float).eps

    # Create W and H
    #avg = np.sqrt(X.mean() / k)
    
    W = np.abs(np.random.uniform(size=(rows, k)))
    #W = avg * np.maximum(W, eps)
    W = np.maximum(W, eps)
    W = np.divide(W, k*W.max())

    H = np.abs(np.random.uniform(size=(k, columns)))
    #H = avg * np.maximum(H, eps)
    H = np.maximum(H, eps)
    H = np.divide(H, k*H.max())

    # Create a Mask
    M = X > 0.0

    # H = H .* (W' * (V ./ (W*H))) ./ sum(W',2)
    # W = W .* ((V ./ (W*H)) * H') ./ sum(H',1)

    for _ in range(n):
        H = H * (W.T @ (X / (W@H))) / np.sum(W.T, axis = 1)
        H = np.maximum(H, eps)
        
        W = W * ((X / (W@H)) @ H.T) / np.sum(H.T, axis = 0)
        W = np.maximum(W, eps)
        
        #Xr = M * (W @ H)
        #cost = np.sum(MX * np.log(MX/ Xr) - MX + Xr)
        #if cost <= l:
        #    break
    
    return W, H
