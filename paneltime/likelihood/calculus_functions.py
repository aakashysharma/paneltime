#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from .. import functions as fu


def dd_func_lags_mult(panel, ll, g, AMAL, vname1, vname2, transpose=False, u_gradient=False):
    de2_zeta_xi_RE, de2_zeta_xi = dd_func_lags_mult_arima(
        panel, ll, g, AMAL, vname1, vname2, transpose, u_gradient)
    dd_re_variance = dd_func_re_variance(
        panel, ll, g, vname1, vname2, de2_zeta_xi_RE, u_gradient)
    d2LL_d2var_zeta_xi, d2LL_d2e_zeta_xi_RE = dd_func_garch(
        panel, ll, g, vname1, vname2, de2_zeta_xi_RE, de2_zeta_xi, dd_re_variance, u_gradient)

    return d2LL_d2var_zeta_xi, d2LL_d2e_zeta_xi_RE


def dd_func_lags_mult_arima(panel, ll, g, AMAL, vname1, vname2, transpose, u_gradient):
    # de_xi is "N x T x m", de_zeta is "N x T x k" and L is "T x T"
    de_xi = g.__dict__['de_'+vname1+'_RE']
    de_zeta = g.__dict__['de_'+vname2+'_RE']
    de2_zeta_xi_RE = None
    if de_xi is None or de_zeta is None:
        return None, None
    (N, T, m) = de_xi.shape
    (N, T, k) = de_zeta.shape
    # ARIMA:
    if not AMAL is None:
        if u_gradient:
            # "T x N x s x m #for error beta-rho covariance, the u gradient must be used
            de2_zeta_xi = -panel.arma_dot.dot(AMAL, g.X_RE, ll)
        else:
            de2_zeta_xi = panel.arma_dot.dot(
                AMAL, de_zeta, ll)  # "T x N x s x m
        if transpose:  # only happens if lags==k
            de2_zeta_xi = de2_zeta_xi + \
                np.swapaxes(de2_zeta_xi, 2, 3)  # adds the transpose
        de2_zeta_xi = de2_zeta_xi*panel.included[4]
    else:
        de2_zeta_xi = None
    de2_zeta_xi_RE = de2_zeta_xi

    return de2_zeta_xi_RE, de2_zeta_xi


def dd_func_re_variance(panel, ll, g, vname1, vname2, de2_zeta_xi_RE, u_gradient):
    if panel.N <= 1 or panel.options.fixed_random_group_eff.value == 0:
        return None
    # voaltility RE:

    de_xi_RE = g.__dict__['de_'+vname1+'_RE']
    de_zeta_RE = g.__dict__['de_'+vname2+'_RE']
    if de_xi_RE is None or de_zeta_RE is None:
        return None
    (N, T, m) = de_xi_RE.shape
    (N, T, k) = de_zeta_RE.shape
    de_xi_RE_r = de_xi_RE.reshape((N, T, m, 1))
    de_zeta_RE_r = de_zeta_RE.reshape((N, T, 1, k))
    incl = panel.included[4]

    dvRE_xi = g.__dict__['dvRE_'+vname1].reshape((N, T, m, 1))
    dvRE_zeta = g.__dict__['dvRE_'+vname2].reshape((N, T, 1, k))

    if not de2_zeta_xi_RE is None:
        d_xi_input = g.__dict__['d_'+vname1+'_input']
        d_zeta_input = g.__dict__['d_'+vname2+'_input']
        try:
            dd_e_RE_sq = (2*de_xi_RE_r*de_zeta_RE_r+2 *
                          ll.e_RE.reshape(N, T, 1, 1)*de2_zeta_xi_RE)*incl
        except:
            dd_e_RE_sq = (2*de_xi_RE_r*de_zeta_RE_r+2 *
                          ll.e_RE.reshape(N, T, 1, 1)*de2_zeta_xi_RE)*incl
        ddmeane2 = panel.mean(dd_e_RE_sq, (0, 1))*incl
        dd_input = (dd_e_RE_sq-ddmeane2)*incl

        ddvRE_d_xi_zeta = ddmeane2-add((ll.re_obj_i_v.ddRE(dd_input, d_xi_input, d_zeta_input, ll.varRE_input, vname1, vname2, panel),
                                        ll.re_obj_t_v.ddRE(dd_input, d_xi_input, d_zeta_input, ll.varRE_input, vname1, vname2, panel)), True)
    else:
        ddvRE_d_xi_zeta = None

    dvarRE,		ddvarRE = ll.dvarRE.reshape(
        (N, T, 1, 1)),  ll.ddvarRE.reshape((N, T, 1, 1))
    dd_re_variance = add(
        (prod((dvarRE, ddvRE_d_xi_zeta)), ddvarRE*dvRE_xi*dvRE_zeta))
    return dd_re_variance


def dd_func_garch(panel, ll, g, vname1, vname2, de2_zeta_xi_RE, de2_zeta_xi, dd_re_variance, u_gradient):
    # GARCH:
    incl = panel.included[4]
    if (g.__dict__['de_'+vname1+'_RE'] is None) or (g.__dict__['de_'+vname2+'_RE'] is None):
        return None, None
    (N, T, m) = g.__dict__['de_'+vname1+'_RE'].shape
    (N, T, k) = g.__dict__['de_'+vname2+'_RE'].shape
    DLL_e = g.DLL_e.reshape(N, T, 1, 1)

    d2LL_d2e_zeta_xi_RE = None
    if not de2_zeta_xi_RE is None:
        d2LL_d2e_zeta_xi_RE = de2_zeta_xi_RE * DLL_e
        d2LL_d2e_zeta_xi_RE = np.sum(np.sum(d2LL_d2e_zeta_xi_RE*incl, 0), 0)

    RE_suffix = ''

    de_xi_RE = g.__dict__['de_'+vname1+'_RE'+RE_suffix]
    de_zeta_RE = g.__dict__['de_'+vname2+'_RE'+RE_suffix]

    dLL_var = g.dLL_var.reshape(N, T, 1, 1)
    if de2_zeta_xi_RE is None:
        de2_zeta_xi_RE = 0
    d2var_zeta_xi = None
    d2LL_d2var_zeta_xi = None
    d_omega_e = None
    d2var_zeta_xi_h = None
    if panel.pqdkm[4] > 0:
        if u_gradient:
            de_zeta_RE = g.__dict__['de_'+vname2+'_RE'+RE_suffix]
        h_e_de2_zeta_xi = ll.h_e_val.reshape(N, T, 1, 1) * de2_zeta_xi_RE
        h_2e_dezeta_dexi = ll.h_2e_val.reshape(
            N, T, 1, 1) * de_xi_RE.reshape((N, T, m, 1)) * de_zeta_RE.reshape((N, T, 1, k))

        d2var_zeta_xi_h = (h_e_de2_zeta_xi + h_2e_dezeta_dexi)

        d2var_zeta_xi_h = panel.arma_dot.dot(ll.GAR_1MA, d2var_zeta_xi_h, ll)

    d2var_zeta_xi = add((d2var_zeta_xi_h,  dd_re_variance, d_omega_e), True)
    if not d2var_zeta_xi is None:
        d2LL_d2var_zeta_xi = np.sum(
            prod((d2var_zeta_xi, dLL_var*incl), True), (0, 1))

    return d2LL_d2var_zeta_xi, d2LL_d2e_zeta_xi_RE


def dd_func_lags(panel, ll, L, d, dLL, transpose=False):
    # d is "N x T x m" and L is "k x T x T"
    if panel.pqdkm[4] == 0:
        return None
    if d is None:
        return None
    (N, T, m) = d.shape
    if L is None:
        x = 0
    elif len(L) == 0:
        return None
    elif type(L) == tuple:  # elif len(L.shape)==3:
        x = panel.arma_dot.dot(L, d, ll)  # "T x N x k x m"
        if x is None:
            return None
    elif len(L.shape) == 2:
        x = fu.dot(L, d).reshape(N, T, 1, m)
    dLL = dLL.reshape((N, T, 1, 1))
    return np.sum(np.sum(dLL*x, 1), 0)  # and sum it


def add(iterable, ignore=False):
    """Sums iterable. If ignore=True all elements except those that are None are added. If ignore=False, None is returned if any element is None. """
    x = None
    for i in range(len(iterable)):
        if not iterable[i] is None:
            if x is None:
                x = iterable[i]
            else:
                x = x+iterable[i]
        else:
            if not ignore:
                return None
    return x


def prod(iterable, ignore=False):
    """Takes the product sum of iterable. If ignore=True all elements except those that are None are multiplied. 
    If ignore=False, None is returned if any element is None. """
    x = None
    for i in iterable:
        if not i is None:
            if x is None:
                x = i
            else:
                x = x*i
        else:
            if not ignore:
                return None
    return x


def sumNT(nparray):
    if nparray is None:
        return None
    s = nparray.shape
    if len(s) < 3:
        raise RuntimeError("Not enough dimensions")

    return np.sum(nparray.reshape(list(s)+[1]), (0, 1))


def concat_matrix(block_matrix):
    m = []
    for i in range(len(block_matrix)):
        r = block_matrix[i]
        C = []
        for j in range(len(r)):
            if not r[j] is None:
                C.append(r[j])
        if len(C):
            m.append(np.concatenate(C, 1))
    m = np.concatenate(m, 0)
    return m


def concat_marray(matrix_array):
    arr = []
    for i in matrix_array:
        if not i is None:
            arr.append(i)
    arr = np.concatenate(arr, 2)
    return arr


def dd_func(d2LL_de2, d2LL_dln_de, d2LL_dln2, de_dh, de_dg, dln_dh, dln_dg, dLL_de2_dh_dg, dLL_dln2_dh_dg):
    a = []
    a.append(dd_func_mult(de_dh, d2LL_de2, de_dg))

    a.append(dd_func_mult(de_dh, d2LL_dln_de, dln_dg))
    a.append(dd_func_mult(dln_dh, d2LL_dln_de, de_dg))

    a.append(dd_func_mult(dln_dh, d2LL_dln2, dln_dg))

    a.append(dLL_de2_dh_dg)
    a.append(dLL_dln2_dh_dg)
    return add(a, True)


def dd_func_mult(d0, mult, d1):
    # d0 is N x T x k and d1 is N x T x m
    if d0 is None or d1 is None or mult is None:
        return None
    (N, T, k) = d0.shape
    (N, T, m) = d1.shape
    if np.any(np.isnan(d0)) or np.any(np.isnan(d1)):
        x = np.empty((k, m))
        x[:] = np.nan
        return x
    d0 = d0*mult
    d0 = np.reshape(d0, (N, T, k, 1))
    d1 = np.reshape(d1, (N, T, 1, m))
    try:
        x = np.sum(np.sum(d0*d1, 0), 0)  # ->k x m
    except RuntimeWarning as e:
        if e.args[0] == 'overflow encountered in multiply':
            d0 = np.minimum(np.maximum(d0, -1e+100), 1e+100)
            d1 = np.minimum(np.maximum(d1, -1e+100), 1e+100)
            x = np.sum(np.sum(d0*d1, 0), 0)  # ->k x m
        else:
            raise RuntimeWarning(e)
    return x
