#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This module contains statistical procedures

from .. import random_effects
from . import stat_dist
from .. import functions as fu
import mpmath as mp
import numpy as np


def var_decomposition(XXNorm=None, X=None):
    """Variance decomposition. Returns the matrix of condition indexes for each factor (rows) and each variable
    (columns). Calculates the normalized sum of squares using square_and_norm if XXNorm is not supplied"""
    if XXNorm is None:
        XXNorm = square_and_norm(X)
    ub = len(XXNorm)
    ev, p = np.linalg.eig(XXNorm)
    if np.any(np.round(ev.imag, 15) != len(ev)*[0]):
        pass
        # print( "non-real XX matrix")

    ev = ev.real
    p = p.real
    d = np.abs(ev)**0.5+1e-100
    MaxEv = np.max(d)
    fi = np.abs(p*p/((d*d).reshape((1, ub))+1E-200))
    fiTot = np.sum(fi, 1).reshape((ub, 1))
    pi = fi/(fiTot + (fiTot == 0)*1E-200)
    pi = pi.T
    CondIx = MaxEv/d
    ind = np.argsort(CondIx)
    pi = pi[ind]
    CondIx = CondIx[ind]
    CondIx = CondIx.reshape((len(CondIx), 1))
    return CondIx, pi, ev, p


def square_and_norm(X):
    """Squares X, and normalize to unit lenght.
    Similar to a correlation matrix, except the
    means are not subtracted"""
    N, T, k = X.shape
    Sumsq = np.sqrt(np.sum(np.sum(X**2, 0), 0))
    Sumsq.resize((k, 1))
    Sumsq = Sumsq*Sumsq.T
    norm = fu.dot(X, X)/(Sumsq+1e-200)
    return norm


def singular_elim(panel, X):
    """Eliminates variables that cause singularity"""
    N, T, k = X.shape
    r = np.arange(k)
    ci_threshold = 50
    keep, XXCorrel = find_singulars(panel, X)
    XXNorm = square_and_norm(X)
    cond_ix, pi, d, p = var_decomposition(XXNorm)
    if max(cond_ix) < ci_threshold:
        return keep, cond_ix
    for cix in range(1, len(cond_ix)):
        if (np.sum(pi[-cix] > 0.5) > 1) and cond_ix[-cix][0] > ci_threshold:
            keep[pi[:, -cix] > 0.5] = False
    return keep, cond_ix


def find_singulars(panel, X):
    """Returns a list with True for variables that cause singularity and False otherwise.
    for the main regression, singularity is detected by the constraints module"""
    N, T, k = X.shape
    XXCorrel = correl(X, panel)
    keep = np.all(np.isnan(XXCorrel) == False, 0)
    keep = keep*np.all((np.abs(np.triu(XXCorrel, 1)) > 0.99) == False, 0)
    x_dev = deviation(panel, X)
    var_x = np.sum(np.sum(x_dev**2, 0), 0)
    keep = keep*(var_x > 0)  # remove constants
    keep[0] = True  # allways keep the first constant term
    return keep, XXCorrel


def adf_test(panel, ll, p):
    """Returns the augmented dickey fuller test statistic and critical value"""
    N, T, k = panel.X.shape
    y = ll.Y_st
    yl1 = roll(y, 1, 1)
    dy = y-yl1
    date_var = np.arange(T).reshape((T, 1))*panel.included[3]  # date count
    X = np.concatenate((panel.included[3], date_var, yl1), 2)
    dyL = []
    for i in range(p):
        dyL.append(roll(dy, i+1, 1))
    dyL = np.concatenate(dyL, 2)
    date_var = (date_var > p+1)
    X = np.concatenate((X, dyL), 2)
    X = X*date_var
    dy = dy*date_var
    X[:, 0:panel.lost_obs+10] = 0
    keep, c_ix = singular_elim(panel, X)
    if not np.all(keep[0:3]):
        return 'NA', 'NA', 'NA'
    beta, se_robust, se = OLS(
        panel, X[:, :, keep], dy, robust_se_lags=10, c=date_var)
    if se_robust[0][0] is None:
        return None, None, None
    if se_robust[2] <= 0:
        return None, None, None
    adf_stat = beta[2]/se_robust[2]
    critval = adf_crit_values(panel.NT, True)
    res = np.append(adf_stat, critval)
    return res


def roll(x, shift, dim):
    b = np.roll(x, shift, dim)
    b[:, :shift] = 0
    return b


def goodness_of_fit(ll, standarized, panel):
    if (panel.NT-panel.args.n_args-1) <= 0:
        return 0, 0, None, None, None, None
    if standarized:
        s_res = panel.var(ll.e_RE)
        s_tot = panel.var(ll.Y_st)
    else:
        s_res = panel.var(ll.u)
        s_tot = panel.var(panel.Y)
    k = max(panel.input.X.shape[1]-panel.input.has_intercept, 1)
    F = (
        ((s_tot-s_res)/k)
        /
        (s_res/(panel.NT-k))
    )
    p = stat_dist.fcdf(F, k, panel.NT-k)
    F_p = None
    if not p is None:
        F_p = 1-stat_dist.fcdf(F, k, panel.NT-k)

    r_unexpl = s_res/s_tot
    Rsq = 1-r_unexpl
    Rsqadj = 1-r_unexpl*(panel.NT-1)/(panel.NT-panel.args.n_args-1)
    panel.args.create_null_ll(panel)
    LL_ratio_OLS = 2*(ll.LL-panel.args.LL_OLS)
    LL_ratio = 2*(ll.LL-panel.args.LL_null)
    return Rsq, Rsqadj, LL_ratio, LL_ratio_OLS, F, F_p


def DurbinWatson(panel, ll):
    """returns a statistic describing how much atocorrelation there is"""
    X = ll.XIV_st
    N, T, k = X.shape
    e = ll.e_RE_norm_centered
    c = panel.included[3][:, :-1]
    DW = np.sum((c*(e[:, 1:]-e[:, :-1]))**2)/np.sum((e*panel.included[3])**2)

    e = ll.u_long
    DW_no_panel = np.sum(((e[1:]-e[:-1]))**2)/np.sum((e)**2)

    return DW, DW_no_panel


def correlogram(panel, e, lags, center=False):
    N, T, k = e.shape
    v = panel.var(e)
    if center:
        e = e-panel.mean(e)
    rho = np.zeros(lags+1)
    rho[0] = 1
    for i in range(1, lags+1):
        a = panel.T_i-i-1 > 0
        incl = (a*panel.included[3])[:, i:, :]
        df = np.sum(incl)
        if df > 0:
            rho[i] = np.sum(incl*e[:, i:]*e[:, 0:-i])/(v*df)
        rho[i] = 0
    return rho  # The probability of no AC given H0 of AC.


def adf_crit_values(n, trend):
    """Returns 1 and 5 percent critical values respectively. Interpolated from table in 
    Fuller, W. A. (1976). Introduction to Statistical Time Series. New York: John Wiley and Sons. ISBN 0-471-28715-6
    table is available at https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test"""
    if trend:
        d = {25: np.array([-3.75, -3.00]), 50: np.array([-3.58, -2.93]), 100: np.array([-3.51, -2.89]),
             250: np.array([-3.46, -2.88]), 500: np.array([-3.44, -2.87]), 10000: np.array([-3.43, -2.86])}
    else:
        d = {25: np.array([-4.38, -3.60]), 50: np.array([-4.15, -3.50]), 100: np.array([-4.04, -3.45]),
             250: np.array([-3.99, -3.43]), 500: np.array([-3.98, -3.42]), 10000: np.array([-3.96, -3.41])}

    if n < 25:
        print(
            "Warning: ADF critical values are not available for fewer than 25 observations")
        return (0, 0)
    k = (25, 50, 100, 250, 500, 10000)
    r = None
    for i in range(len(k)-1):
        if n >= k[i] and n < 500:
            r = d[k[i]]+(n-k[i])*((d[k[i+1]]-d[k[i]]) /
                                  (k[i+1]-k[i]))  # interpolation
            return r
    if r is None:
        return d[10000]


def correl(X, panel=None, covar=False):
    """Returns the correlation of X. Assumes three dimensional matrices. """
    if not panel is None:
        X = X*panel.included[3]
        N, T, k = X.shape
        N = panel.NT
        mean = np.sum(np.sum(X, 0), 0).reshape((1, k))/N
    else:
        N, k = X.shape
        mean = np.sum(X, 0).reshape((1, k))/N
    cov = fu.dot(X, X)/N

    cov = cov-(mean.T*mean)

    if covar:
        return cov

    stdx = (np.diag(cov)**0.5).reshape((1, k))
    stdx = (stdx.T*stdx)
    stdx[np.isnan(stdx)] = 0
    corr = (stdx > 0)*cov/(stdx+(stdx == 0)*1e-100)
    corr[stdx <= 0] = 0

    return corr


def deviation(panel, X):
    N, T, k = X.shape
    x = X*panel.included[3]
    mean = np.sum(np.sum(x, 0), 0).reshape((1, 1, k))/panel.NT
    return (X-mean)*panel.included[3]


def correl_2dim(X, Y=None, covar=False):
    """Returns the correlation of X and Y. Assumes two dimensional matrixes. If Y is not supplied, the 
    correlation matrix of X is returned"""
    if type(X) == list:
        X = Concat(X)
    single = Y is None
    if single:
        Y = X
    T, k = X.shape
    T, m = Y.shape
    X_dev = X-np.mean(X, 0)
    Y_dev = Y-np.mean(Y, 0)
    cov = np.dot(X_dev.T, Y_dev)
    if covar:
        return cov/(len(X)-1)
    stdx = np.sum(X_dev**2, 0).reshape((1, k))**0.5
    if single:
        stdy = stdx
    else:
        stdy = np.sum(Y_dev**2, 0).reshape((1, k))**0.5
    std_matr = stdx.T*stdy
    std_matr = std_matr+(std_matr == 0)*1e-200
    corr = cov/std_matr
    if corr.shape == (1, 1):
        corr = corr[0][0]
    return corr


def get_singular_list(panel, XX):
    a, b = singular_elim(panel, XX)
    if len(panel.args.names_v) != len(XX):
        names = [None]*len(XX)
    else:
        names = np.array(panel.args.names_v)[a == False]
    idx = np.array(range(len(a)))[a == False]
    s = ', '.join([f"{names[i]}" for i in range(len(idx))])
    return s


def OLS(panel, X, Y, add_const=False, return_rsq=False, return_e=False, c=None, robust_se_lags=0):
    """runs OLS after adding const as the last variable"""
    if c is None:
        c = panel.included[3]
    N, T, k = X.shape
    NT = panel.NT
    if add_const:
        X = np.concatenate((c, X), 2)
        k = k+1
    X = X*c
    Y = Y*c
    XX = fu.dot(X, X)
    XY = fu.dot(X, Y)
    try:
        beta = np.linalg.solve(XX, XY)
    except np.linalg.LinAlgError:
        s = get_singular_list(panel, X)
        print("The following variables caused singularity runtime and must be removed: "+s)
        beta = np.zeros(k).reshape((k, 1))
        if return_rsq:
            return beta, 0
        elif return_e:
            beta, Y
        elif robust_se_lags:
            nonarr = np.array([None]*k).reshape((k, 1))
            return beta, nonarr, nonarr
        return beta
    if return_rsq or return_e or robust_se_lags:
        e = (Y-fu.dot(X, beta))*c
        if return_rsq:
            v0 = panel.var(e, included=c)
            v1 = panel.var(Y, included=c)
            Rsq = 1-v0/v1
            # Rsqadj=1-(v0/v1)*(NT-1)/(NT-k-1)
            return beta, Rsq
        elif return_e:
            return beta, e*c
        elif robust_se_lags:
            XXInv = np.linalg.inv(XX)
            se_robust, se, V = robust_se(panel, robust_se_lags, XXInv, X*e)
            return beta, se_robust.reshape(k, 1), se.reshape(k, 1)
    return beta


def OLS_simple(Y, X, addconst=False, residuals=True):
    """Returns the OLS residuals if residuals. For use with two dimiensional arrays only"""
    if addconst:
        n = len(X)
        X = np.concatenate((np.ones((n, 1)), X), 1)
    XY = np.dot(X.T, Y)
    XX = np.dot(X.T, X)
    XXInv = np.linalg.inv(XX)
    beta = np.dot(XXInv, XY)
    e = Y-np.dot(X, beta)
    if residuals:
        return beta, e
    else:
        return beta


def newey_west_wghts(L, XErr):
    """Calculates the Newey-West autocorrelation consistent weighting matrix. Either err_vec or XErr is required"""
    N, T, k = XErr.shape
    S = np.zeros((k, k))
    try:
        a = min(L, T)
    except:
        a = 0
    for i in range(1, min(L, T)):
        w = 1-(i+1)/(L)
        XX = fu.dot(XErr[:, i:], XErr[:, 0:T-i])
        S += w*(XX+XX.T)
    return S


def robust_cluster_weights(panel, XErr, cluster_dim, whites):
    """Calculates the Newey-West autocorrelation consistent weighting matrix. Either err_vec or XErr is required"""
    N, T, k = XErr.shape
    if cluster_dim == 0:  # group cluster
        if N <= 1:
            return 0
        mean = panel.mean(XErr, 0)
    elif cluster_dim == 1:  # time cluster
        mean = random_effects.mean_time(panel, XErr, True)
        T, m, k = mean.shape
        mean = mean.reshape((T, k))
    S = fu.dot(mean.T, mean)-whites
    return S


def robust_se(panel, L, hessin, XErr, nw_only=True):
    """Returns the maximum robust standard errors considering all combinations of sums of different combinations
    of clusters and newy-west"""
    w, W = sandwich_var(hessin, fu.dot(XErr, XErr))  # whites
    nw, NW = sandwich_var(hessin, newey_west_wghts(L, XErr))  # newy-west
    if panel.N > 1:
        c0, C0 = sandwich_var(hessin, robust_cluster_weights(
            panel, XErr, 0, w))  # cluster dim 1
        c1, C1 = sandwich_var(hessin, robust_cluster_weights(
            panel, XErr, 1, w))  # cluster dim 2
    else:
        c0, c1, C0, C1 = 0, 0, 0*W, 0*W
    v = np.array([
        nw,
        nw+c0,
        nw+c1,
        nw+c1+c0,
        w*0
    ])
    V = np.array([
        NW,
        NW+C0,
        NW+C1,
        NW+C1+C0,
        W*0
    ])
    V = V+W
    s = np.max(w+v, 0)
    se_robust = np.maximum(s, 0)**0.5
    i = np.argmax(np.sum(w+v, 1))
    se_std = np.maximum(w, 0)**0.5
    return se_robust, se_std, V[i]


def sandwich_var(hessin, V):
    hessinV = np.dot(hessin, V)
    V = np.dot(hessinV, hessin)
    v = np.diag(V)
    return v, V


def breusch_godfrey_test(panel, ll, lags):
    """returns the probability that err_vec are not auto correlated"""
    e = ll.e_RE_norm_centered
    X = ll.XIV_st
    N, T, k = X.shape
    if 2*lags > T-5:
        lags = int(0.5*max((0, T-5)))
    X_u = X[:, lags:T]
    u = e[:, lags:T]
    c = panel.included[3][:, lags:T]
    for i in range(1, lags+1):
        X_u = np.append(X_u, e[:, lags-i:T-i], 2)
    Beta, Rsq = OLS(panel, X_u, u, False, True, c=c)
    T = (panel.NT-k-1-lags)
    BGStat = T*Rsq
    rho = Beta[k:]
    if T > 1:
        ProbNoAC = 1.0-stat_dist.chisq(BGStat, lags)
    else:
        ProbNoAC = 1.0
    return ProbNoAC, rho, Rsq  # The probability of no AC given H0 of AC.


def JB_normality_test(e, panel):
    """Jarque-Bera test for normality. 
    returns the probability that a set of residuals are drawn from a normal distribution"""

    df = len(e)
    mu1 = np.mean(e)
    e = e-mu1
    s = (np.sum(e**2)/df)**0.5
    mu3 = np.sum(e**3)/df
    mu4 = np.sum(e**4)/df
    S = mu3/s**3
    K = mu4/s**4
    JB = df*((S**2)+0.25*(K-3)**2)/6.0
    p = 1.0-stat_dist.chisq(JB, 2)

    omn, omn_sign = Omnibus(S, K-3, df)

    return p, JB, S, K, omn, omn_sign


def Omnibus(g1, g2, n):
    "D'Agostino's K^2"
    # https://en.wikipedia.org/wiki/D%27Agostino%27s_K-squared_test

    mu2 = 6*(n-2)/((n+1)*(n+3))
    gamma2 = 36*(n-7)*(n**2 + 2*n - 5)/((n-2)*(n+5)*(n+7)*(n+9))

    W = ((2*gamma2+4)**0.5-1)**0.5
    if W <= 1:
        return None, None
    delta = 1/np.log(W)**0.5
    alpha = (2/(W**2-1))**0.5

    Z1 = delta*float(mp.asinh(g1/(alpha*mu2**0.5)))

    mu1 = -6/(n+1)
    mu2 = 24*n*(n-1)*(n-3)/((n+1)**2*(n+3)*(n+5))
    gamma1 = 6*(n**2-5*n+2)
    gamma1 *= (6*(n+3)*(n+5)/(n*(n-2)*(n-3)))**0.5
    gamma1 = gamma1/((n+7)*(n+9))

    A = 6 + (8/gamma1)*((2/gamma1)+(1+4/gamma1**2)**0.5)
    try:
        Z2 = (9*A/2)**0.5*(1-(2/(9*A)) -
                           ((1-2/A)/(1 + ((g2-mu1)/mu2**0.5)*(2/(A-4)**0.5)))**(1/3))
    except:
        return None, None

    Omnibus = Z1**2 + Z2**2
    sign = 1-stat_dist.chisq(Omnibus, 2)
    return Omnibus, sign
