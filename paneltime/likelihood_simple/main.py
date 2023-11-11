#!/usr/bin/env python
# -*- coding: utf-8 -*-

# contains the log likelihood object

import time
import traceback
import numpy as np
from ..output import stat_functions
from .. import functions as fu
from . import function
from ..output import stat_dist
from ..processing import model_parser


from pathlib import Path
import os
import numpy.ctypeslib as npct
import ctypes as ct
p = os.path.join(Path(__file__).parent.absolute(), 'cfunctions')
if os.name == 'nt':
    cfunct = npct.load_library('ctypes.dll', p)
else:
    cfunct = npct.load_library('ctypes.so', p)


CDPT = ct.POINTER(ct.c_double)
CIPT = ct.POINTER(ct.c_uint)


class LL:
    """Calculates the log likelihood given arguments arg (either in dictonary or array form), and creates an object
    that store dynamic variables that depend on the \n
    If args is a dictionary, the ARMA-GARCH orders are 
    determined from the dictionary. If args is a vector, the ARMA-GARCH order needs to be consistent
    with the  panel object
    """

    def __init__(self, args, panel, constraints=None, print_err=False):
        self.err_msg = ''
        self.errmsg_h = ''

        self.args = panel.args.create_args(args, panel, constraints)
        self.h_err = ""
        self.LL = None
        # self.LL=self.LL_calc(panel)
        try:
            self.LL = self.LL_calc(panel)
            if np.isnan(self.LL):
                self.LL = None
        except Exception as e:
            if print_err:
                traceback.print_exc()
                print(self.errmsg_h)

    def LL_calc(self, panel):
        X = panel.XIV
        N, T, k = X.shape
        incl = panel.included[3]
        self.set_var_bounds(panel)

        G = fu.dot(panel.W_a, self.args.args_d['omega'])
        if 'initvar' in self.args.args_d:
            G[0, 0, 0] += abs(self.args.args_d['initvar'][0])
        else:
            G[0, 0, 0] += panel.args.init_var

        # Idea for IV: calculate Z*u throughout. Mazimize total sum of LL.
        u = panel.Y-fu.dot(X, self.args.args_d['beta'])

        matrices = self.arma_calc(panel, u*incl, self.h_add, G)
        if matrices is None:
            return None
        AMA_1, AMA_1AR, GAR_1, GAR_1MA, e, var, h = matrices

        # NOTE: self.h_val itself is also set in ctypes.cpp/ctypes.c. If you change self.h_val below, you need to
        # change it in the c-scripts too. self.h_val must be calcualted below as well for later calulcations.
        if panel.options.EGARCH.value == 0:
            e_sq = (e**2+(e == 0)*1e-18)
            nd = 1
            self.h_val, self.h_e_val, self.h_2e_val = (
                e**2+self.h_add)*incl, nd*2*e*incl, nd*2*incl
            # possibility of additional parameter in sq res function
            self.h_z_val, self.h_2z_val,  self.h_ez_val = None, None, None
        else:
            minesq = 1e-20
            e_sq = np.maximum(e**2, minesq)
            nd = e**2 > minesq

            self.h_val, self.h_e_val, self.h_2e_val = np.log(e_sq+self.h_add)*incl, 2*incl*e/(
                e_sq+self.h_add), incl*2/(e_sq+self.h_add) - incl*2*e**2/(e_sq+self.h_add)**2
            # possibility of additional parameter in sq res function
            self.h_z_val, self.h_2z_val,  self.h_ez_val = None, None, None

        if False:  # debug
            from .. import debug
            if np.any(h != self.h_val):
                print(
                    'the h calculated in the c function and the self.h_val calcualted here do not match')
            debug.test_c_armas(u, var, e, panel, self, G)

        LL_full, v, v_inv, self.dvar_pos = function.LL(
            panel, var, e_sq, e, self.minvar, self.maxvar)
        self.tobit(panel, LL_full)
        LL = np.sum(LL_full*incl)
        self.LL_all = np.sum(LL_full)
        self.add_variables(panel, matrices, u, var, v,
                           G, e, e_sq, v_inv, LL_full)
        if abs(LL) > 1e+100:
            return None
        return LL

    def set_var_bounds(self, panel):
        if panel.options.EGARCH.value == 0:
            self.minvar = 0.01*panel.args.init_var
            self.maxvar = 1000*panel.args.init_var
            self.h_add = panel.args.init_var
        else:
            self.minvar = -100
            self.maxvar = 100
            self.h_add = 0.1

    def add_variables(self, panel, matrices, u, var, v, G, e, e_sq, v_inv, LL_full):
        self.v_inv05 = v_inv**0.5
        self.e_norm = e*self.v_inv05
        self.e_RE_norm_centered = (
            self.e_norm-panel.mean(self.e_norm))*panel.included[3]
        self.u = u
        self.var,  self.v,    self.LL_full = var,       v,    LL_full
        self.G = G
        self.e = e
        self.e_sq = e_sq
        self.v_inv = v_inv

    def tobit(self, panel, LL):
        if sum(panel.tobit_active) == 0:
            return
        g = [1, -1]
        self.F = [None, None]
        for i in [0, 1]:
            if panel.tobit_active[i]:
                I = panel.tobit_I[i]
                self.F[i] = stat_dist.norm(g[i]*self.e_norm[I])
                LL[I] = np.log(self.F[i])

    def standardize(self, panel, reverse_difference=False):
        """Adds X and Y and error terms after ARIMA-E-GARCH transformation and random effects to self. 
        If reverse_difference and the ARIMA difference term d>0, the standardized variables are converted to
        the original undifferenced order. This may be usefull if the predicted values should be used in another 
        differenced regression."""
        if hasattr(self, 'Y_st'):
            return
        m = panel.lost_obs
        N, T, k = panel.X.shape
        if model_parser.DEFAULT_INTERCEPT_NAME in panel.args.caption_d['beta']:
            m = self.args.args_d['beta'][0, 0]
        else:
            m = panel.mean(panel.Y)
        # e_norm=self.standardize_variable(panel,self.u,reverse_difference)
        self.Y_st,   self.Y_st_long = self.standardize_variable(
            panel, panel.Y, reverse_difference)
        self.X_st,   self.X_st_long = self.standardize_variable(
            panel, panel.X, reverse_difference)
        self.XIV_st, self.XIV_st_long = self.standardize_variable(
            panel, panel.XIV, reverse_difference)
        self.Y_pred_st = fu.dot(self.X_st, self.args.args_d['beta'])
        self.Y_pred = fu.dot(panel.X, self.args.args_d['beta'])
        self.e_norm_long = self.stretch_variable(panel, self.e_norm)
        self.Y_pred_st_long = self.stretch_variable(panel, self.Y_pred_st)
        self.Y_pred_long = np.dot(panel.input.X, self.args.args_d['beta'])
        self.u_long = panel.input.Y-self.Y_pred_long

        Rsq, Rsqadj, LL_ratio, LL_ratio_OLS = stat_functions.goodness_of_fit(
            self, False, panel)
        Rsq2, Rsqadj2, LL_ratio2, LL_ratio_OLS2 = stat_functions.goodness_of_fit(
            self, True, panel)
        a = 0

    def standardize_variable(self, panel, X, norm=False, reverse_difference=False):
        X = panel.arma_dot.dot(self.AMA_1AR, X, self)
        if (not panel.Ld_inv is None) and reverse_difference:
            X = fu.dot(panel.Ld_inv, X)*panel.a[3]
        if norm:
            X = X*self.v_inv05
        X_long = self.stretch_variable(panel, X)
        return X, X_long

    def stretch_variable(self, panel, X):
        N, T, k = X.shape
        m = panel.map
        NT = panel.total_obs
        X_long = np.zeros((NT, k))
        X_long[m] = X
        return X_long

    def copy_args_d(self):
        return copy_array_dict(self.args.args_d)

    def h(self, panel, e, z):
        return h(e, z, panel)

    def arma_calc(self, panel, u, h_add, G):
        matrices = set_garch_arch(panel, self.args.args_d, u, h_add, G)
        if matrices is None:
            return None
        self.AMA_1, self.AMA_1AR, self.GAR_1, self.GAR_1MA, self.e, self.var, self.h = matrices
        self.AMA_dict = {'AMA_1': None, 'AMA_1AR': None,
                         'GAR_1': None, 'GAR_1MA': None}
        return matrices

    def predict(self, W, W_next=None):
        d = self.args.args_d
        self.u_pred = pred_u(self.u, self.e, d['rho'], d['lambda'])
        u_pred = pred_u(self.u[:, :-1], self.e[:, :-1],
                        d['rho'], d['lambda'], self.e[:, -1])  # test
        self.var_pred = pred_var(
            self.h, self.var, d['psi'], d['gamma'], d['omega'], W_next, self.minvar, self.maxvar)
        var_pred = pred_var(self.h[:, :-1], self.var[:, :-1], d['psi'],
                            d['gamma'], d['omega'], W, self.minvar, self.maxvar)  # test

        return {'predicted residual': self.u_pred, 'predicted variance': self.var_pred}


def pred_u(u, e, rho, lmbda, e_now=0):
    if len(lmbda) == 0 and len(rho) == 0:
        return 0
    u_pred = e_now
    if len(rho):
        u_pred += np.sum([
            rho[i]*u[:, -i-1] for i in range(len(rho))
        ], 1)
    if len(lmbda):
        u_pred += np.sum([
            lmbda[i]*e[:, -i-1] for i in range(len(lmbda))
        ], 1)

    return u_pred[0, 0]


def pred_var(h, var, psi, gamma, omega, W, minvar, maxvar):
    W = test_variance_signal(W, h, omega)
    if W is None:
        G = omega[0, 0]
    else:
        G = np.dot(W, omega)
    a, b = 0, 0
    if len(psi):
        a = sum([
            psi[i]*h[:, -i-1] for i in range(len(psi))
        ])
    if len(gamma):
        b = sum([
            gamma[i]*(var[:, -i-1]) for i in range(len(gamma))
        ])

    var_pred = G + a + b
    var_pred = max(min((var_pred[0, 0], maxvar)), minvar)

    return var_pred


def test_variance_signal(W, h, omega):
    if W is None:
        return None
    N, T, k = h.shape
    if N == 1:
        W = W.flatten()
        if len(W) != len(omega):
            raise RuntimeError("The variance signals needs to be a numpy array of numbers with "
                               "the size equal to the HF-argument variables +1, and first variable must be 1")
        return W.reshape((1, len(omega)))
    else:
        try:
            NW, kW = W.shape
            if NW != N or kW != k:
                raise RuntimeError("Rows and columns in variance signals must correspond with"
                                   "the number of groups and the size equal to the number of "
                                   "HF-argument variables +1, respectively")
        except:
            raise RuntimeError("If there are more than one group, the variance signal must be a matrix with"
                               "Rows and columns in variance signals must correspond with"
                               "the number of groups and the size equal to the number of "
                               "HF-argument variables +1, respectively")
    return W


def set_garch_arch(panel, args, u, h_add, G):
    """Solves X*a=b for a where X is a banded matrix with 1 or zero, and args along
    the diagonal band"""
    N, T, _ = u.shape
    rho = np.insert(-args['rho'], 0, 1)
    psi = args['psi']
    psi = np.insert(args['psi'], 0, 0)

    AMA_1, AMA_1AR, GAR_1, GAR_1MA, e, var, h = (
        np.append([1], np.zeros(T-1)),
        np.zeros(T),
        np.append([1], np.zeros(T-1)),
        np.zeros(T),
        np.zeros((N, T, 1)),
        np.zeros((N, T, 1)),
        np.zeros((N, T, 1))
    )

    lmbda = args['lambda']
    gmma = -args['gamma']

    parameters = np.array((N, T,
                           len(lmbda), len(rho), len(gmma), len(psi),
                           panel.options.EGARCH.value, panel.tot_lost_obs,
                           h_add))

    cfunct.armas(parameters.ctypes.data_as(CIPT),
                 lmbda.ctypes.data_as(CDPT), rho.ctypes.data_as(CDPT),
                 gmma.ctypes.data_as(CDPT), psi.ctypes.data_as(CDPT),
                 AMA_1.ctypes.data_as(CDPT), AMA_1AR.ctypes.data_as(CDPT),
                 GAR_1.ctypes.data_as(CDPT), GAR_1MA.ctypes.data_as(CDPT),
                 u.ctypes.data_as(CDPT),
                 e.ctypes.data_as(CDPT),
                 var.ctypes.data_as(CDPT),
                 h.ctypes.data_as(CDPT),
                 G.ctypes.data_as(CDPT)
                 )

    r = []
    # Creating nympy arrays with name properties.
    for i in ['AMA_1', 'AMA_1AR', 'GAR_1', 'GAR_1MA']:
        r.append((locals()[i], i))
    for i in ['e', 'var', 'h']:
        r.append(locals()[i])

    return r


def set_garch_arch_scipy(panel, args):
    # after implementing ctypes, the scipy version might be dropped entirely
    p, q, d, k, m = panel.pqdkm
    nW, n = panel.nW, panel.max_T

    AAR = -lag_matr(-panel.I, args['rho'])
    AMA_1AR, AMA_1 = solve_mult(args['lambda'], AAR, panel.I)
    if AMA_1AR is None:
        return
    GMA = lag_matr(panel.I*0, args['psi'])
    GAR_1MA, GAR_1 = solve_mult(-args['gamma'], GMA, panel.I)
    if GAR_1MA is None:
        return
    r = []
    for i in ['AMA_1', 'AMA_1AR', 'GAR_1', 'GAR_1MA']:
        r.append((locals()[i], i))
    return r


def solve_mult(args, b, I):
    """Solves X*a=b for a where X is a banded matrix with 1  and args along
    the diagonal band"""
    import scipy
    n = len(b)
    q = len(args)
    X = np.zeros((q+1, n))
    X[0, :] = 1
    X2 = np.zeros((n, n))
    w = np.zeros(n)
    r = np.arange(n)
    for i in range(q):
        X[i+1, :n-i-1] = args[i]
    try:
        X_1 = scipy.linalg.solve_banded((q, 0), X, I)
        if np.any(np.isnan(X_1)):
            return None, None
        X_1b = fu.dot(X_1, b)
    except:
        return None, None

    return X_1b, X_1


def add_to_matrices(X_1, X_1b, a, ab, r):
    for i in range(0, len(a)):
        if i > 0:
            d = (r[i:], r[:-i])
            X_1[d] = a[i]
        else:
            d = (r, r)
        X_1b[d] = ab[i]
    return X_1, X_1b


def lag_matr(L, args):
    k = len(args)
    if k == 0:
        return L
    L = 1*L
    r = np.arange(len(L))
    for i in range(k):
        d = (r[i+1:], r[:-i-1])
        if i == 0:
            d = (r, r)
        L[d] = args[i]

    return L


def copy_array_dict(d):
    r = dict()
    for i in d:
        r[i] = np.array(d[i])
    return r
