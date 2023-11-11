#!/usr/bin/env python
# -*- coding: utf-8 -*-


from .. import functions as fu
from . import calculus_functions as cf
from . import function
import numpy as np
import time


def arima_grad(k, x, ll, sign, pre, panel):
    if k == 0:
        return None
    (N, T, m) = x.shape
    x = panel.arma_dot.dotroll(pre, k, sign, x, ll)
    x.resize(N, T, k)
    extr_value = 1e+100
    if np.max(np.abs(x)) > extr_value:
        x[np.abs(x) > extr_value] = np.sign(
            x[np.abs(x) > extr_value])*extr_value
    return x*panel.a[3]


def garch_arima_grad(ll, d, varname, panel):
    panel = panel
    dvar_sigma = None
    if panel.pqdkm[4] > 0 and not d is None:  # eqs. 33-34
        ((N, T, k)) = d.shape
        x = cf.prod((ll.h_e_val, d))
        dvar_sigma = panel.arma_dot.dot(ll.GAR_1MA, x, ll)
    return dvar_sigma


def gradient(ll, panel, DLL_e=None, dLL_var=None, return_G=False):

    incl = panel.included[3]

    u, e, h_e_val, var, h_val, v = ll.u, ll.e, ll.h_e_val, ll.var, ll.h_val, ll.v
    p, q, d, k, m = panel.pqdkm

    if DLL_e is None:
        dLL_var, DLL_e = function.gradient(ll, panel)
    # ARIMA:
    de_rho = arima_grad(p, u*incl, ll, -1, ll.AMA_1, panel)
    de_lambda = arima_grad(q, e*incl, ll, -1, ll.AMA_1, panel)
    de_beta = -panel.arma_dot.dot(ll.AMA_1AR, panel.XIV*incl, ll)*panel.a[3]

    (de_rho, de_lambda, de_beta) = (de_rho, de_lambda, de_beta)

    dvar_sigma_rho = garch_arima_grad(ll, de_rho,  'rho', panel)
    dvar_sigma_lambda = garch_arima_grad(ll, de_lambda, 'lambda', panel)
    dvar_sigma_beta = garch_arima_grad(ll, de_beta, 'beta', panel)

    # GARCH:

    dvar_omega = panel.arma_dot.dot(ll.GAR_1, panel.W_a, ll)
    dvar_initvar = None
    if 'initvar' in ll.args.args_d:
        dvar_initvar = ll.GAR_1[0].reshape((1, panel.max_T, 1))
        if not panel.options.EGARCH.value and ll.args.args_d['initvar'][0] < 0:
            dvar_initvar = -dvar_initvar

    (dvar_gamma, dvar_psi, dvar_mu, dvar_z_G,
     dvar_z) = (None, None, None, None, None)

    if panel.N > 1:
        dvar_mu = cf.prod((ll.dvarRE_mu, incl))
    else:
        dvar_mu = None

    if m > 0:
        dvar_gamma = arima_grad(k, var, ll, 1, ll.GAR_1, panel)
        dvar_psi = arima_grad(m, h_val, ll, 1, ll.GAR_1, panel)
        if not ll.h_z_val is None:
            dvar_z_G = fu.dot(ll.GAR_1MA, ll.h_z_val)
            (N, T, k) = dvar_z_G.shape

        dvar_z = dvar_z_G

    (dvar_gamma, dvar_psi, dvar_mu, dvar_z_G, dvar_z) = (
        dvar_gamma, dvar_psi, dvar_mu, dvar_z_G, dvar_z)

    # LL

    # final derivatives:
    dLL_beta = cf.add((cf.prod((dvar_sigma_beta, dLL_var)),
                      cf.prod((de_beta, DLL_e))), True)
    dLL_rho = cf.add((cf.prod((dvar_sigma_rho, dLL_var)),
                     cf.prod((de_rho, DLL_e))), True)
    dLL_lambda = cf.add(
        (cf.prod((dvar_sigma_lambda, dLL_var)), cf.prod((de_lambda, DLL_e))), True)
    dLL_gamma = cf.prod((dvar_gamma, dLL_var))
    dLL_psi = cf.prod((dvar_psi, dLL_var))
    dLL_omega = cf.prod((dvar_omega, dLL_var))
    dLL_initvar = cf.prod((dvar_initvar, dLL_var))
    dLL_mu = cf.prod((dvar_mu, dLL_var))
    dLL_z = cf.prod((dvar_z, dLL_var))

    G = cf.concat_marray((dLL_beta, dLL_rho, dLL_lambda, dLL_gamma,
                         dLL_psi, dLL_omega, dLL_initvar, dLL_mu, dLL_z))
    g = np.sum(G, (0, 1))
    # For debugging:
    # from .. import debug
    # print(debug.grad_debug(ll,panel,0.00001))
    # print(g)
    # if np.sum((g-gn)**2)>10000000:
    # a=0
    # print(gn)
    # a=debug.grad_debug_detail(ll, panel, 0.00000001, 'LL', 'beta',0)
    # dLLeREn,deREn=debug.LL_calc_custom(ll, panel, 0.0000001)

    if return_G:
        return g, G
    else:
        return g
