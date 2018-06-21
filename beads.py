"""

Baseline estimation and denoising using sparsity (BEADS)

INPUT
    y: Noisy observation
    d: Filter order (d = 1 or 2)
    fc: Filter cut-off frequency (cycles/sample) (0 < fc < 0.5)
    r: Asymmetry ratio
    lam0, lam1, lam2: Regularization parameters

OUTPUT
    x: Estimated sparse-derivative signal
    f: Estimated baseline
    cost: Cost function history

Reference:
    Chromatogram baseline estimation and denoising using sparsity (BEADS)
    Xiaoran Ning, Ivan W. Selesnick, Laurent Duval
    Chemometrics and Intelligent Laboratory Systems (2014)
    doi: 10.1016/j.chemolab.2014.09.014
    Available online 30 September 2014

Author:
    Hisao, Chun-Yi

Github:


"""
import numpy as np
from scipy.sparse import spdiags


def beads(y, d, fc, r, lam0, lam1, lam2):
    # The following parameter may be altered.
    Nit = 3  # Nit: Number of iterations
    pen = 'L1_v2'  # pen : penalty function for sparse derivative ('L1_v1' or 'L1_v2')
    EPS0 = 1e-6  # cost smoothing parameter for x (small positive value)
    EPS1 = 1e-6  # cost smoothing parameter for derivatives(small positive value)

    if pen is 'L1_v1':
        phi = lambda xx: np.sqrt(np.power(abs(xx), 2) + EPS1)
        wfun = lambda xx: 1. / np.sqrt(np.power(abs(xx), 2) + EPS1)
    elif pen is 'L1_v2':
        phi = lambda xx: abs(xx) - EPS1 * np.log(abs(xx) + EPS1)
        wfun = lambda xx: 1. / (abs(xx) + EPS1)
    else:
        print('penalty must be L1_v1, L1_v2')
        x, cost, f = [], [], []
        return x, cost, f

    #  equation (25)
    theta = lambda xx: sum(xx[(xx>EPS0)]) - r * sum(xx[(xx<-EPS0)]) + sum((1+r)/(4*EPS0) * np.power(xx[(abs(xx)<=EPS0)], 2) + (1-r)/2 * xx[(abs(xx)<=EPS0)] + EPS0*(1+r)/4)
    # theta = @(x) sum(x(x>EPS0)) - r * sum(x(x<-EPS0)) + sum( (1+r)/(4*EPS0)*x(abs(x)<=EPS0).^2 + (1-r)/2 * x(abs(x)<=EPS0) + EPS0*(1+r)/4 )

    N = len(y)
    y = np.reshape(a=y, newshape=[N, 1])
    x = y
    cost = np.array([])
    A, B = BAfilt(d, fc, N)
    H = lambda xx: np.dot(B, (linv(A, xx)))  # H = @(x) B*(A\x)
    e = np.ones((N-1, 1))
    # ee1 = np.array([-e, e]).squeeze()
    # ee2 = np.array([e, -2*e, e]).squeeze()
    d1 = spdiags(np.array([-e, e]).squeeze(), np.array([0, 1]), N-1, N)  # D1 = spdiags([-e e], [0 1], N-1, N)
    d2 = spdiags(np.array([e, -2*e, e]).squeeze(), np.arange(0, 3), N-2, N)  # D2 = spdiags([e -2*e e], 0:2, N-2, N)
    D1, D2 = d1.A, d2.A
    D1[-1, -1], D2[-1, -1] = 1., 1.  # lack the last one item [-1, -1]
    D = np.vstack((D1, D2))  # D = [D1;  D2];
    BTB = np.dot(np.transpose(B), B)

    w = np.vstack(([lam1 * np.ones((N-1, 1)), lam2 * np.ones((N-2, 1))]))
    b = (1-r) / 2 * np.ones((N, 1))
    d = np.dot(BTB, (linv(A, y))) - lam0 * np.dot(np.transpose(A), b)

    gamma = np.ones((N, 1))
    for i in range(1, Nit+1):
        print('step: ', i)
        wf = wfun(np.dot(D, x))
        wff = w * wf
        lmda = spdiags(wff.transpose(), 0, 2 * N - 3, 2 * N - 3)
        Lmda = lmda.A

        k = np.array(abs(x) > EPS0)  # return index 1d
        gamma[~k] = ((1 + r) / 4) / abs(EPS0)
        gamma[k] = ((1 + r) / 4) / abs(x[k])
        Gamma = spdiags(gamma.transpose(), 0, N, N)

        M = 2 * lam0 * Gamma.A + np.dot(np.dot(np.transpose(D), Lmda), D).transpose()
        x = np.dot(A, (linv(BTB + np.dot(np.dot(np.transpose(A), M), A), d)))


        cost = np.append(cost,  0.5*sum(np.power(abs(H(xx=y-x)), 2))
                         + lam0 * theta(x) + lam1 * sum(phi(np.diff(x.squeeze()))) + lam2 * sum(phi(np.diff(x.squeeze(), 2))))
        # cost(i) = 0.5 * sum(abs(H(y - x)). ^ 2) + lam0 * theta(x) + lam1 * sum(phi(diff(x))) + lam2 * sum(phi(diff(x, 2)));
        pass

    f = y - x - H(y - x)

    return x, cost, f


def BAfilt(d, fc, N):
    """
     --- local function ----

    function [A, B] = BAfilt(d, fc, N)
     [A, B] = BAfilt(d, fc, N)

     Banded matrices for zero-phase high-pass filter.
     The matrices are 'sparse' data type in MATLAB.

     INPUT
       d  : degree of filter is 2d (use d = 1 or 2)
       fc : cut-off frequency (normalized frequency, 0 < fc < 0.5)
       N  : length of signal
    """

    b1 = [1, -1]
    for i in range(1, d):  # for i = 1:d - 1
        b1 = np.convolve(a=b1, v=[-1, 2, -1])
    pass

    b = np.convolve(a=b1, v=[-1, 1]) #  b = conv(b1, [-1 1])

    omc = 2 * np.pi * fc
    t = np.power(((1 - np.cos(omc)) / (1 + np.cos(omc))), d)

    a = 1
    for i in range(1, d+1):  # for i = 1:d
        a = np.convolve(a=a, v=[1, 2, 1])
    pass
    a = b + t * a
    # print(a, b)
    xa, xb = (a*np.ones((N, 1))).transpose(), (b*np.ones((N, 1))).transpose()
    dr = np.arange(-d, d+1)
    A = spdiags(xa, dr, N, N)  # A: Symmetric banded matrix
    B = spdiags(xb, dr, N, N)  # B: banded matrix
    return A.A, B.A
    pass


# left inverse
def linv(a, b):
    c, resid, rank, s = np.linalg.lstsq(a, b)
    # a_linv = lin.solve(a.T.dot(a), a.T)
    # c = a_linv.dot(b)
    return c


def test():

    #####
    # test signal
    t = np.linspace(-1, 1, 200)
    sig = np.cos(2*np.pi*7*t) + np.sin(3*np.pi*7*t+300*t)

    fc = 1 / 200
    d = 1
    r = 6
    amp = 0.8
    lam0 = 0.5 * amp
    lam1 = 5 * amp
    lam2 = 4 * amp

    x, f, cost = beads(y=sig, d=d, fc=fc, r=r, lam0=lam0, lam1=lam1, lam2=lam2)
    print(x)


if __name__ == '__main__':
    test()
