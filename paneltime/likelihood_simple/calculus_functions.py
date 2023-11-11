#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from .. import functions as fu


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
