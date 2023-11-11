#!/usr/bin/env python
# -*- coding: utf-8 -*-

# contains the log likelihood object

from ..output import stat_functions
from .. import random_effects as re
from .. import functions as fu
from . import function
from ..output import stat_dist
from ..processing import model_parser
from . import arma

import numpy as np
import traceback
import time


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

        # checking settings. If the FE/RE is done on the data before LL
        gfre = panel.options.fixed_random_group_eff.value
        tfre = panel.options.fixed_random_time_eff.value
        vfre = panel.options.fixed_random_variance_eff.value

        self.re_obj_i = re.re_obj(panel, True, panel.T_i, panel.T_i, gfre)
        self.re_obj_t = re.re_obj(
            panel, False, panel.date_count_mtrx, panel.date_count, tfre)
        self.re_obj_i_v = re.re_obj(
            panel, True, panel.T_i, panel.T_i, gfre*vfre)
        self.re_obj_t_v = re.re_obj(
            panel, False, panel.date_count_mtrx, panel.date_count, tfre*vfre)

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
            G[0, 0, 0] = abs(self.args.args_d['initvar'][0])
        else:
            G[0, 0, 0] = panel.args.init_var

        # Idea for IV: calculate Z*u throughout. Mazimize total sum of LL.
        u = panel.Y-fu.dot(X, self.args.args_d['beta'])
        u_RE = (u+self.re_obj_i.RE(u, panel)+self.re_obj_t.RE(u, panel))*incl

        matrices = self.arma_calc(panel, u_RE, self.h_add, G)
        if matrices is None:
            return None
        AMA_1, AMA_1AR, GAR_1, GAR_1MA, e_RE, var, h = matrices

        # NOTE: self.h_val itself is also set in ctypes.cpp/ctypes.c. If you change self.h_val below, you need to
        # change it in the c-scripts too. self.h_val must be calcualted below as well for later calulcations.
        if panel.options.EGARCH.value == 0:
            e_REsq = (e_RE**2+(e_RE == 0)*1e-18)
            nd = 1
            self.h_val, self.h_e_val, self.h_2e_val = (
                e_RE**2+self.h_add)*incl, nd*2*e_RE*incl, nd*2*incl
            # possibility of additional parameter in sq res function
            self.h_z_val, self.h_2z_val,  self.h_ez_val = None, None, None
        else:
            minesq = 1e-20
            e_REsq = np.maximum(e_RE**2, minesq)
            nd = e_RE**2 > minesq

            self.h_val, self.h_e_val, self.h_2e_val = np.log(e_REsq+self.h_add)*incl, 2*incl*e_RE/(
                e_REsq+self.h_add), incl*2/(e_REsq+self.h_add) - incl*2*e_RE**2/(e_REsq+self.h_add)**2
            # possibility of additional parameter in sq res function
            self.h_z_val, self.h_2z_val,  self.h_ez_val = None, None, None

        self.variance_RE(panel, e_REsq)
        if False:  # debug
            from .. import debug
            if np.any(h != self.h_val):
                print(
                    'the h calculated in the c function and the self.h_val calcualted here do not match')
            debug.test_c_armas(u_RE, var, e_RE, panel, self, G)

        LL_full, v, v_inv, self.dvar_pos = function.LL(
            panel, var, e_REsq, e_RE, self.minvar, self.maxvar)
        self.tobit(panel, LL_full)
        LL = np.sum(LL_full*incl)
        self.LL_all = np.sum(LL_full)
        self.add_variables(panel, matrices, u, u_RE, var, v,
                           G, e_RE, e_REsq, v_inv, LL_full)
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

    def add_variables(self, panel, matrices, u, u_RE, var, v, G, e_RE, e_REsq, v_inv, LL_full):
        self.v_inv05 = v_inv**0.5
        self.e_norm = e_RE*self.v_inv05
        self.e_RE_norm_centered = (
            self.e_norm-panel.mean(self.e_norm))*panel.included[3]
        self.u, self.u_RE = u,  u_RE
        self.var,  self.v,    self.LL_full = var,       v,    LL_full
        self.G = G
        self.e_RE = e_RE
        self.e_REsq = e_REsq
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

    def variance_RE(self, panel, e_REsq):
        """Calculates random/fixed effects for variance."""
        # not in use, expermental. Should be applied to normalize before ARIMA/GARCH
        self.vRE, self.varRE, self.dvarRE = panel.zeros[3], panel.zeros[3], panel.zeros[3]
        self.ddvarRE, self.dvarRE_mu, self.ddvarRE_mu_vRE = panel.zeros[3], None, None
        self.varRE_input, self.ddvarRE_input, self.dvarRE_input = None, None, None
        return
        if panel.options.fixed_random_variance_eff.value == 0:
            return panel.zeros[3]
        if panel.N == 0:
            return None

        meane2 = panel.mean(e_REsq)
        self.varRE_input = (e_REsq-meane2)*panel.included[3]

        mine2 = 0
        mu = panel.options.variance_RE_norm.value
        self.vRE_i = self.re_obj_i_v.RE(self.varRE_input, panel)
        self.vRE_t = self.re_obj_t_v.RE(self.varRE_input, panel)
        self.meane2 = meane2
        vRE = meane2*panel.included[3]-self.vRE_i-self.vRE_t
        self.vRE = vRE
        small = vRE <= mine2
        big = small == False
        vREbig = vRE[big]
        vREsmall = vRE[small]

        varREbig = np.log(vREbig+mu)
        varREsmall = (np.log(mine2+mu)+((vREsmall-mine2)/(mine2+mu)))
        varRE, dvarRE, ddvarRE = np.zeros(vRE.shape), np.zeros(
            vRE.shape), np.zeros(vRE.shape)

        varRE[big] = varREbig
        varRE[small] = varREsmall
        self.varRE = varRE*panel.included[3]

        dvarRE[big] = 1/(vREbig+mu)
        dvarRE[small] = 1/(mine2+mu)
        self.dvarRE = dvarRE*panel.included[3]

        ddvarRE[big] = -1/(vREbig+mu)**2
        self.ddvarRE = ddvarRE*panel.included[3]

        return self.varRE

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
        self.Y_long = panel.input.Y
        self.X_long = panel.input.X
        self.Y_st,   self.Y_st_long = self.standardize_variable(
            panel, panel.Y, reverse_difference)
        self.X_st,   self.X_st_long = self.standardize_variable(
            panel, panel.X, reverse_difference)
        self.XIV_st, self.XIV_st_long = self.standardize_variable(
            panel, panel.XIV, reverse_difference)
        self.Y_pred_st = fu.dot(self.X_st, self.args.args_d['beta'])
        self.Y_pred = fu.dot(panel.X, self.args.args_d['beta'])
        self.e_RE_norm_centered_long = self.stretch_variable(
            panel, self.e_RE_norm_centered)
        self.Y_pred_st_long = self.stretch_variable(panel, self.Y_pred_st)
        self.Y_pred_long = np.dot(panel.input.X, self.args.args_d['beta'])
        self.u_long = np.array(panel.input.Y-self.Y_pred_long)

        a = 0

    def standardize_variable(self, panel, X, norm=False, reverse_difference=False):
        X = panel.arma_dot.dot(self.AMA_1AR, X, self)
        X = (X+self.re_obj_i.RE(X, panel, False) +
             self.re_obj_t.RE(X, panel, False))
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

    def arma_calc(self, panel, u, h_add, G):
        matrices = arma.set_garch_arch(panel, self.args.args_d, u, h_add, G)
        self.AMA_1, self.AMA_1AR, self.GAR_1, self.GAR_1MA, self.e, self.var, self.h = matrices
        self.AMA_dict = {'AMA_1': None, 'AMA_1AR': None,
                         'GAR_1': None, 'GAR_1MA': None}
        return matrices

    def predict(self, W, W_next=None):
        d = self.args.args_d
        self.u_pred = pred_u(self.u, self.e, d['rho'], d['lambda'])
        u_pred = pred_u(self.u[:, :-1], self.e[:, :-1],
                        d['rho'], d['lambda'], 0)  # test
        self.var_pred = pred_var(
            self.h, self.var, d['psi'], d['gamma'], d['omega'], W_next, self.minvar, self.maxvar)
        # var_pred = pred_var(self.h[:,:-1], self.var[:,:-1], d['psi'], d['gamma'], d['omega'], W, self.minvar, self.maxvar)#test
        if not hasattr(self, 'Y_pred'):
            self.standardize()

        return {'predicted residual': self.u_pred, 'predicted variance': self.var_pred,
                'in-sample predicted Y': self.Y_pred_long, 'in-sample predicted variance': self.v}


def pred_u(u, e, rho, lmbda, e_now=0):
    if len(lmbda) == 0 and len(rho) == 0:
        return 0
    u_pred = e_now
    if len(rho):
        u_pred += sum([
            rho[i]*u[:, -i-1] for i in range(len(rho))
        ])
    if len(lmbda):
        u_pred += sum([
            lmbda[i]*e[:, -i-1] for i in range(len(lmbda))
        ])
    if len(u_pred) == 1:
        u_pred = u_pred[0, 0]
    return u_pred


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
    var_pred = np.maximum(np.minimum(var_pred, maxvar), minvar)
    try:
        if len(var_pred) == 1:
            var_pred = var_pred[0, 0]
    except TypeError:
        # var_pred is not iterable, which is expected with GARCH(0,0)
        pass
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


def copy_array_dict(d):
    r = dict()
    for i in d:
        r[i] = np.array(d[i])
    return r
