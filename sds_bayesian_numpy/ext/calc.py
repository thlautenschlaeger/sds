import numpy as np
from autograd.scipy.linalg import block_diag
from time import time

def logdotexp(A, B):
    """ Computes the dot product of two matrices that are
     in log space
     :param A: matrix or vector in log space
     :param B: matrix or vector in log space
    """
    assert A.shape[0] == B.shape[0], \
        "Shapes of A: {} and B: {} on axis 0 do not match".format(A.shape[0], B.shape[0])

    max_A = np.max(A)
    max_B = np.max(B)
    C = np.exp(A - max_A) @ np.exp(B - max_B)
    C = np.log(C)
    C += max_A + max_B

    return C

def _weighted_lin_reg(X, Y, w, fit_intercept=False):
    """
    :param X: input x [TxD]
    :param Y: output y [TxD]
    :param w: weight [Tx1]
    :param c: constant [1xD]
    :returns: weight matrix A [DxD] and updated constant vector c [1xD]
    """

    assert X.shape[0] == Y.shape[0] == w.shape[0], \
        "X.shape[0]: {} | Y.shape[0]: {} | w.shape[0]: {} shapes do not match".format(X.shape[0], Y.shape[0], w.shape[0])

    W = np.diag(w)
    tmp1 = np.linalg.inv(X.T @ W @ X)
    tmp2 =  tmp1 @ X.T @ W
    Weight = (tmp2 @ (Y)).T

    if fit_intercept:
        A, c = Weight[:, :-1], Weight[:, -1]
    else:
        A = Weight
        c = 0

    return A, c

def weighted_lin_reg(X, Y, w, fit_intercept=False,
                     mu0=0, sig0=1e32,
                     nu0=1, psi0=1e-32):
    """
    :param X: input x [TxD]
    :param Y: output y [TxD]
    :param w: weight [Tx1]
    :param c: constant [1xD]
    :returns: weight matrix A [DxD] and updated constant vector c [1xD]
    """

    assert X.shape[0] == Y.shape[0] == w.shape[0], \
        "X.shape[0]: {} | Y.shape[0]: {} | w.shape[0]: {} shapes do not match".format(X.shape[0], Y.shape[0], w.shape[0])

    D = X.shape[1] - 1
    P = Y.shape[1]

    # sig0 *= np.eye(D)
    # if fit_intercept:
    #     sig0 = block_diag(sig0, np.eye(1))
    # tmp1 = np.linalg.inv(sig0)

    # W = np.diag(w)
    # tmp1 += X.T @ W @ X
    # tmp2 = X.T @ W @ Y
    # Weight = np.linalg.solve(tmp1, tmp2).T

    W = np.diag(w)
    tmp1 = np.linalg.inv(X.T @ W @ X)
    tmp2 =  tmp1 @ X.T @ W
    Weight = (tmp2 @ (Y)).T

    if fit_intercept:
        A, c = Weight[:, :-1], Weight[:, -1]
    else:
        A = Weight
        c = 0

    nu = nu0
    Psi = psi0 * np.eye(P)

    yhat = np.dot(X[:, :-1], A.T) + c
    resid = Y - yhat
    nu += np.sum(w)
    cov = np.einsum('t,ti,tj->ij', w, resid, resid)
    # tmp2 = np.sum(w[:, None, None] * resid[:, :, None] * resid[:, None, :], axis=0)
    # assert np.allclose(tmp1, tmp2)
    Psi += cov

    # Get MAP estimate of posterior covariance
    Sigma = Psi / (nu + P + 1)

    return A, c, Sigma









