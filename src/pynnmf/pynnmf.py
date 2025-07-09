# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import logging
import numpy as np


logger = logging.getLogger(__name__)


def rwnmf(X, k, alpha=0.1, tol_fit_improvement=1e-4, tol_fit_error=1e-4, num_iter=1000, seed=None):
    if isinstance(seed, int):
        np.random.seed(seed)

    # applies regularized weighted nmf to matrix X with k factors
    # ||X-UV^T||
    eps = np.finfo(float).eps
    early_stop = False

    # get observations matrix W
    W = np.isnan(X)
    # print('W')
    # print(W)
    X[W] = 0  # set missing entries as 0
    # print(X)
    W = ~W
    # print('~W')
    # print(W)
    #W = X > 0.0

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
        error = np.linalg.norm(X - Xr, ord='fro')

    return Xr, U, V, error


def cost_fb(A, B, M=None):
    if M is None:
        if np.any(np.isnan(A)):
            M = np.isnan(A)
            A[M] = 0
            M = ~M
            cost = np.linalg.norm((M*A) - (M*B), ord='fro')
            A[~M] = np.nan
        else:
            M = A > 0.0
            cost = np.linalg.norm((M*A) - (M*B), ord='fro')
    else:
        cost = np.linalg.norm((M*A) - (M*B), ord='fro')
    return cost


def nmf_mu(X, k, n=1000, l=1E-3, seed=None):
    if isinstance(seed, int):
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
    #M = X > 0.0
    M = np.isnan(X)
    X[M] = 0
    M = ~M

    for _ in range(n):
        W = np.multiply(W, np.divide(
            (M*X)@H.T-l*np.linalg.norm(W, 'fro'), (M*(W@H))@H.T))
        W = np.maximum(W, eps)
        H = np.multiply(H, np.divide(
            W.T@(M*X)-l*np.linalg.norm(H, 'fro'), W.T@(M*(W@H))))
        H = np.maximum(H, eps)

        Xr = W @ H
        cost = cost_fb(X, Xr, M)
        if cost <= l:
            break

    X[~M] = np.nan
    return Xr, W, H, cost


# Kullback–Leibler
def cost_kl(A, B, M=None):
    if M is None:
        if np.any(np.isnan(A)):
            M = np.isnan(A)
            M = ~M
        else:
            M = A > 0.0
    return np.sum(A[M]*np.log(A[M]/B[M])-A[M]+B[M])


def nmf_mu_kl(X, k, n=100, l=1E-3, seed=None, r=20):
    if isinstance(seed, int):
        np.random.seed(seed)
    
    # Create a Mask
    #M = X > 0.0
    M = np.isnan(X)
    X[M] = 0
    M = ~M

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

    if seed is None:
        Xr = W @ H
        cost = cost_kl(X, Xr, M)

        for _ in range(r):
            Wt = np.abs(np.random.uniform(size=(rows, k)))
            #W = avg * np.maximum(W, eps)
            Wt = np.maximum(Wt, eps)
            Wt = np.divide(Wt, k*Wt.max())

            Ht = np.abs(np.random.uniform(size=(k, columns)))
            #H = avg * np.maximum(H, eps)
            Ht = np.maximum(Ht, eps)
            Ht = np.divide(Ht, k*Ht.max())

            Xr = Wt @ Ht
            cost_temp = cost_kl(X, Xr, M)

            if cost_temp < cost:
                W = Wt
                H = Ht
                cost = cost_temp

    for _ in range(n):
        I = np.where(X==0, W@H, X)
        H = H * (W.T @ (I / (W@H)) / np.sum(W.T, axis=1)[:,None])
        H = np.maximum(H, eps)

        I = np.where(X==0, W@H, X)
        W = W * ((I / (W@H) @ H.T) / np.sum(H.T, axis=0))
        W = np.maximum(W, eps)

        Xr = W @ H
        cost = cost_kl(X, Xr)
        if cost <= l:
            break

    X[~M] = np.nan
    return Xr, W, H, cost


# Itakura-Saito
def cost_is(A, B, M=None):
    if M is None:
        if np.any(np.isnan(A)):
            M = np.isnan(A)
            A[M] = 0
            M = ~M
            cost = np.sum((A[M]/B[M])-np.log(A[M]/B[M])-1)
            A[~M] = np.nan
        else:
            M = A > 0.0
            cost = np.sum((A[M]/B[M])-np.log(A[M]/B[M])-1)
    else:
        cost = np.sum((A[M]/B[M])-np.log(A[M]/B[M])-1)
    return cost


def nmf_mu_is(X, k, n=100, l=1E-3, seed=None, r=20):
    if isinstance(seed, int):
        np.random.seed(seed)
    
    # Create a Mask
    #M = X > 0.0
    M = np.isnan(X)
    X[M] = 0
    M = ~M

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

    if seed is None:
        Xr = W @ H
        cost = cost_is(X, Xr, M)

        for _ in range(r):
            Wt = np.abs(np.random.uniform(size=(rows, k)))
            #W = avg * np.maximum(W, eps)
            Wt = np.maximum(Wt, eps)
            Wt = np.divide(Wt, k*Wt.max())

            Ht = np.abs(np.random.uniform(size=(k, columns)))
            #H = avg * np.maximum(H, eps)
            Ht = np.maximum(Ht, eps)
            Ht = np.divide(Ht, k*Ht.max())

            Xr = Wt @ Ht
            cost_temp = cost_is(X, Xr, M)

            if cost_temp < cost:
                W = Wt
                H = Ht
                cost = cost_temp

    # W = W .* sqrt.(((V ./ (W*H))*(H./sum(W*H,1))') ./(sum((H./sum(W*H,1))',1)))
    # H = H .* sqrt.(((W ./ sum(W*H,2))' * (V./(W*H)) ./ (sum((W ./ sum(W*H,2))',2))))

    for _ in range(n):
        I = np.where(X==0, W@H, X)
        #H = H * (W.T @ (I / (W@H)) / np.sum(W.T, axis=1)[:,None])
        H = H * np.sqrt(((W / np.sum(W@H, axis=1)[:,None]).T @ (I / (W@H)) / (np.sum((W / np.sum(W@H, axis=1)[:,None]).T, axis=1)[:,None])))
        H = np.maximum(H, eps)

        I = np.where(X==0, W@H, X)
        #W = W * ((I / (W@H) @ H.T) / np.sum(H.T, axis=0))
        W = W * np.sqrt(((I / (W@H)) @ (H / np.sum(W@H, axis=0)).T) / (np.sum((H / np.sum(W@H, axis=0)).T, axis = 0)))
        W = np.maximum(W, eps)

        Xr = W @ H
        cost = cost_is(X, Xr)
        if cost <= l:
            break

    X[~M] = np.nan
    return Xr, W, H, cost
