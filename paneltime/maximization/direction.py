#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ..output import stat_functions as stat
import numpy as np
import itertools


def get(g, x, H, constr, f, hessin, simple=True):
    n = len(x)
    if simple or (H is None):
        dx = -(np.dot(hessin, g.reshape(n, 1))).flatten()
    else:
        dx, H, applied_constraints = solve(constr, H, g, x, f)
    dx_norm = normalize(dx, x)

    return dx, dx_norm, H


def new(g, x, H, constr, f, dx, alam):
    n = len(x)
    dx, slope, rev = slope_check(g, dx)
    if constr is None:
        return dx*alam, slope, rev, []
    elif len(constr.intervals) == 0:
        return dx*alam, slope, rev, []
    elif constr.within(x + dx):
        return dx*alam, slope, rev, []

    dxalam, H, applied_constraints = solve(constr, H, g*alam, x, f)
    if np.sum(g*dxalam) < 0.0:
        dxalam, H, applied_constraints = solve(constr, H, -g*alam, x, f)
        slope = np.sum(g*dx)
        rev = True
    return dxalam, slope, rev, applied_constraints


def slope_check(g, dx):
    rev = False
    slope = np.sum(g*dx)  # Scale if attempted step is too big.
    if slope <= 0.0:
        dx = -dx
        slope = np.sum(g*dx)
        rev = True
    return dx, slope, rev


def solve(constr, H, g, x, f):
    """Solves a second degree taylor expansion for the dc for df/dc=0 if f is quadratic, given gradient
    g, hessian H, inequalty constraints c and equalitiy constraints c_eq and returns the solution and 
    and index constrained indicating the constrained variables"""

    if H is None:
        raise RuntimeError('Cant solve with no coefficient matrix')
    try:
        list(constr.keys())[0]
    except:
        dx = -np.linalg.solve(H, g)
        return dx, H, []
    H = make_hess_posdef(H)
    H_orig = np.array(H)
    m = len(H)

    n, g, H, delmap, keys, dx, idx = add_constraints(constr, H, g)
    k = len(keys)

    xi_full = np.zeros(m)
    OK = False

    applied_constraints = []
    for j in range(k):
        xi_full[idx] = dx[:n]
        OK = constr.within(x+xi_full, False)
        if OK:
            break
        key = keys[j]
        dx, H = kuhn_tucker(constr, key, j, n, H, g, x, f, dx, delmap, OK)
        applied_constraints.append(key)
    xi_full = np.zeros(m)
    xi_full[idx] = dx[:n]
    H_full = np.zeros((m, m))
    idx = idx.reshape((1, m))
    nz = np.nonzero(idx*idx.T)
    H_full[nz] = H[np.nonzero(np.ones((n, n)))]
    H_full[H_full == 0] = H_orig[H_full == 0]
    return xi_full, H_full, applied_constraints


def add_constraints(constr, H, g):
    # testing the direction in case H is singular (should be handled by constr, but
    # sometimes it isn't), and removing randomly until H is not singular

    idx = np.ones(len(g), dtype=bool)
    idx[list(constr.fixed.keys())] = False
    for i in np.array(list(itertools.product([False, True], repeat=sum(idx)))):
        idx_new = np.array(idx)
        idx_active = np.array(idx[idx])
        idx_active[i] = False
        idx_new[idx] = idx_active
        n, g_new, H_new, delmap, keys = remove_and_enlarge(
            constr, H, g, idx_new)
        try:
            dx = -np.linalg.solve(H_new, g_new)
            break
        except np.linalg.LinAlgError as e:
            if e.args[0] != 'Singular matrix':
                raise np.linalg.LinAlgError(e)
    return n, g_new, H_new, delmap, keys, dx, idx_new


def remove_and_enlarge(constr, H, g, idx):
    m = len(g)
    delmap = np.arange(m)
    if not np.all(idx == True):  # removing fixed constraints from the matrix

        H = H[idx][:, idx]
        g = g[idx]
        delmap -= np.cumsum(idx == False)
        # if for some odd reason, the deleted variables are referenced later, an out-of-bounds error will be thrown
        delmap[idx == False] = m

    n = len(H)
    keys = list(constr.intervals.keys())
    k = len(keys)
    H = np.concatenate((H, np.zeros((n, k))), 1)
    H = np.concatenate((H, np.zeros((k, n+k))), 0)
    g = np.append(g, np.zeros(k))

    for i in range(k):
        H[n+i, n+i] = 1

    return n, g, H, delmap, keys


def kuhn_tucker(constr, key, j, n, H, g, x, f, dx, delmap, OK, recalc=True):
    q = None
    c = constr.intervals[key]
    i = delmap[key]
    if not c.value is None:
        q = -(c.value-x[key])
    elif x[key]+dx[i] < c.min:
        q = -(c.min-x[key])
    elif x[key]+dx[i] > c.max:
        q = -(c.max-x[key])
    if q != None:
        if OK:
            a = 0
        H[i, n+j] = 1
        H[n+j, i] = 1
        H[n+j, n+j] = 0
        g[n+j] = q
        if recalc:
            dx = -np.linalg.solve(H, g)
    return dx, H


def normalize(dx, x):
    x = np.abs(x)
    dx_norm = (x != 0)*dx/(x+(x < 1e-100))
    dx_norm = (x < 1e-2)*dx+(x >= 1e-2)*dx_norm
    return dx_norm


def make_hess_posdef(H):
    c, var_prop, ev, p = stat.var_decomposition(XXNorm=H)
    ev = ((ev < 0) - 1.0 * (ev >= 0)) * ev
    H = np.dot(p, np.dot(np.diag(ev), p.T))
    return H
