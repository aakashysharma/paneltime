#!/usr/bin/env python
# -*- coding: utf-8 -*-

# used for debugging


from . import functions as fu
from .likelihood import main as lgl
import numpy as np
import time
import os


def hess_debug(ll, panel, g, d):
    """Calculate the Hessian nummerically, using the analytical gradient. For debugging. Assumes correct and debugged gradient"""
    x = ll.args.args_v
    n = len(x)
    dx = np.identity(n)*d
    H = np.zeros((n, n))
    ll0 = lgl.LL(x, panel)
    f0 = g.get(ll0)
    for i in range(n):
        ll = lgl.LL(x+dx[i], panel)
        if not ll is None:
            f1 = g.get(ll)
            H[i] = (f1-f0)/d

    return H


def grad_debug(ll, panel, d):
    """Calcualtes the gradient numerically. For debugging"""
    x = ll.args.args_v
    n = len(x)
    dx = np.abs(x.reshape(n, 1))*d
    dx = dx+(dx == 0)*d
    dx = np.identity(n)*dx

    g = np.zeros(n)
    f0 = lgl.LL(x, panel)
    for i in range(n):
        for j in range(5):
            dxi = dx[i]*(0.5**j)
            f1 = lgl.LL(x+dxi, panel)
            if not f1 is None:
                if not f1.LL is None:
                    g[i] = (f1.LL-f0.LL)/dxi[i]
                    break
    return g


def grad_debug_detail(f0, panel, d, llname, varname1, pos1=0):
    args1 = lgl.copy_array_dict(f0.args.args_d)
    args1[varname1][pos1] += d

    f0 = lgl.LL(f0.args.args_d, panel)
    f1 = lgl.LL(args1, panel)

    if type(llname) == list or type(llname) == tuple:
        ddL = (f1.__dict__[llname[0]].__dict__[llname[1]] -
               f0.__dict__[llname[0]].__dict__[llname[1]])/d
    else:
        ddL = (f1.__dict__[llname]-f0.__dict__[llname])/d
    return ddL


def test_c_armas(u_RE, var, e_RE, panel, ll, G):
    var2 = panel.arma_dot.dot(ll.GAR_1, G, ll) + \
        panel.arma_dot.dot(ll.GAR_1MA, ll.h_val, ll)
    e_RE2 = panel.arma_dot.dot(ll.AMA_1AR, u_RE, ll)
    print(f"Testsums arma: c:{np.sum(var**2)}, py:{np.sum(var2**2)}")
    print(f"Testsums e: c:{np.sum(e_RE**2)}, py:{np.sum(e_RE2**2)}")


def hess_debug_detail(f0, panel, d, llname, varname1, varname2, pos1=0, pos2=0):
    args1 = lgl.copy_array_dict(f0.args.args_d)
    args2 = lgl.copy_array_dict(f0.args.args_d)
    args3 = lgl.copy_array_dict(f0.args.args_d)
    args1[varname1][pos1] += d
    args2[varname2][pos2] += d
    args3[varname1][pos1] += d
    args3[varname2][pos2] += d
    f1 = lgl.LL(args1, panel)
    f2 = lgl.LL(args2, panel)
    f3 = lgl.LL(args3, panel)
    if type(llname) == list:
        ddL = (f3.__dict__[llname[0]].__dict__[llname[1]]-f2.__dict__[llname[0]].__dict__[llname[1]]
               - f1.__dict__[llname[0]].__dict__[llname[1]]+f0.__dict__[llname[0]].__dict__[llname[1]])/(d**2)
    else:
        ddL = (f3.__dict__[llname]-f2.__dict__[llname] -
               f1.__dict__[llname]+f0.__dict__[llname])/(d**2)
    return ddL


def LL_calc(self, panel):
    panel = self.panel
    X = panel.XIV
    matrices = set_garch_arch(panel, self.args.args_d)
    if matrices is None:
        return None

    AMA_1, AMA_1AR, GAR_1, GAR_1MA = matrices
    (N, T, k) = X.shape
    # Idea for IV: calculate Z*u throughout. Mazimize total sum of LL.
    u = panel.Y-fu.dot(X, self.args.args_d['beta'])
    e = fu.dot(AMA_1AR, u)
    e_RE = (e+self.re_obj_i.RE(e, panel) +
            self.re_obj_t.RE(e, panel))*panel.included[3]

    e_REsq = (e_RE**2+(e_RE == 0)*1e-18)
    grp = self.variance_RE(panel, e_REsq)  # experimental

    W_omega = fu.dot(panel.W_a, self.args.args_d['omega'])

    lnv_ARMA = self.garch(panel, GAR_1MA, e_RE)

    lnv = W_omega+lnv_ARMA  # 'N x T x k' * 'k x 1' -> 'N x T x 1'
    lnv += grp
    self.dlnv_pos = (lnv < 100)*(lnv > -100)
    lnv = np.maximum(np.minimum(lnv, 100), -100)
    v = np.exp(lnv)*panel.a[3]
    v_inv = np.exp(-lnv)*panel.a[3]

    LL = self.LL_const-0.5*(lnv+(e_REsq)*v_inv)

    self.tobit(panel, LL)
    LL = np.sum(LL*panel.included[3])

    self.add_variables(matrices, u, e, lnv_ARMA, lnv, v,
                       W_omega, grp, e_RE, e_REsq, v_inv)
    if abs(LL) > 1e+100:
        return None
    return LL


def save_reg_data(ll, panel, fname='repr.csv', heading=True):
    # saves data neccessary to reproduce
    N, T, k = panel.X.shape
    N, T, m = panel.W_a.shape
    heading = [f'X{i}' for i in range(k)]
    heading += [f'W{i}' for i in range(m)]
    heading += ['Y', 'e', 'u', 'var', 'LL', 'AMA_1AR',
                'GAR_1MA', 'GAR_1', 'coefs_names', 'coefs']
    a = np.concatenate((panel.X, panel.W_a, panel.Y, ll.e, ll.u, ll.var), 2)
    a = a.reshape((T, a.shape[2]))
    coefs = np.zeros((T, 1))
    coef_names = np.array([['']]*T, dtype='<U20')
    coef_names[:] = ''
    coefs[:len(ll.args.args_v), 0] = ll.args.args_v
    coef_names[:len(ll.args.args_v), 0] = ll.args.names_v
    a = np.concatenate((a,
                        ll.LL_full[0].reshape((T, 1)),
                        ll.AMA_1AR[0].reshape((T, 1))[::-1],
                        ll.GAR_1MA[0].reshape((T, 1))[::-1],
                        ll.GAR_1[0].reshape((T, 1))[::-1],
                        coef_names,
                        coefs
                        ), 1)
    a = np.concatenate(([heading], a), 0)
    np.savetxt(fname, a, fmt='%s', delimiter=';')
