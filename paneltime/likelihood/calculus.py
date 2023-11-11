#!/usr/bin/env python
# -*- coding: utf-8 -*-


from .. import functions as fu
from . import calculus_functions as cf
from . import function
import numpy as np
import time


class gradient:

    def __init__(self, panel, callback):
        self.panel = panel
        self.callback = callback

    def arima_grad(self, k, x, ll, sign, pre):
        if k == 0:
            return None
        (N, T, m) = x.shape
        x = self.panel.arma_dot.dotroll(pre, k, sign, x, ll)
        x = x.reshape(N, T, k)
        extr_value = 1e+100
        if np.max(np.abs(x)) > extr_value:
            x[np.abs(x) > extr_value] = np.sign(
                x[np.abs(x) > extr_value])*extr_value
        return x*self.panel.a[3]

    def garch_arima_grad(self, ll, dRE, varname):
        panel = self.panel
        groupeffect = 0
        groupeffect, dvRE_dx = None, None
        d_input = 0
        if self.panel.N > 1 and panel.options.fixed_random_group_eff.value > 0 and not dRE is None:
            d_eRE_sq = 2*ll.e_RE*dRE
            dmeane2 = panel.mean(d_eRE_sq, (0, 1))
            d_input = (d_eRE_sq-dmeane2)*panel.a[3]
            dvRE_dx = dmeane2*panel.a[3]-ll.re_obj_i_v.dRE(
                d_input, ll.varRE_input, varname, panel)-ll.re_obj_t_v.dRE(d_input, ll.varRE_input, varname, panel)
            groupeffect = ll.dvarRE*dvRE_dx*panel.a[3]

        dvar_sigma = None
        if self.panel.pqdkm[4] > 0 and not dRE is None:  # eqs. 33-34
            ((N, T, k)) = dRE.shape
            x = cf.prod((ll.h_e_val, dRE))
            dvar_sigma = self.panel.arma_dot.dot(ll.GAR_1MA, x, ll)
        dvar_e = cf.add((dvar_sigma, groupeffect), True)
        return dvar_e, dvar_sigma, dvRE_dx, d_input

    def get(self, ll, DLL_e=None, dLL_var=None, return_G=False):

        self.callback(perc=0.05, text='', task='gradient')
        (self.DLL_e, self.dLL_var) = (DLL_e, dLL_var)
        panel = self.panel
        incl = self.panel.included[3]
        re_obj_i, re_obj_t = ll.re_obj_i, ll.re_obj_t
        u, e_RE, u_RE, h_e_val, var, h_val, v = ll.u, ll.e_RE, ll.u_RE, ll.h_e_val, ll.var, ll.h_val, ll.v
        p, q, d, k, m = panel.pqdkm
        nW = panel.nW
        if DLL_e is None:
            dLL_var, DLL_e = function.gradient(ll, self.panel)
        self.X_RE = (panel.XIV+re_obj_i.RE(panel.XIV, panel) +
                     re_obj_t.RE(panel.XIV, panel))*panel.included[3]
        # ARIMA:
        de_rho_RE = self.arima_grad(p, u_RE, ll, -1, ll.AMA_1)
        de_lambda_RE = self.arima_grad(q, e_RE, ll, -1, ll.AMA_1)
        de_beta_RE = - \
            self.panel.arma_dot.dot(ll.AMA_1AR, self.X_RE, ll)*panel.a[3]

        (self.de_rho_RE, self.de_lambda_RE, self.de_beta_RE) = (
            de_rho_RE, de_lambda_RE, de_beta_RE)

        dvar_sigma_rho,		dvar_sigma_rho_G,		dvRE_rho, d_rho_input = self.garch_arima_grad(
            ll, self.de_rho_RE,  'rho')
        dvar_sigma_lambda, 	dvar_sigma_lambda_G,	dvRE_lambda, d_lambda_input = self.garch_arima_grad(
            ll, self.de_lambda_RE, 'lambda')
        dvar_sigma_beta,	dvar_sigma_beta_G,		dvRE_beta, d_beta_input = self.garch_arima_grad(
            ll, self.de_beta_RE, 'beta')

        (self.dvar_sigma_rho, self.dvar_sigma_lambda, self.dvar_sigma_beta) = (
            dvar_sigma_rho, dvar_sigma_lambda, dvar_sigma_beta)
        (self.dvar_sigma_rho_G, self.dvar_sigma_lambda_G, self.dvar_sigma_beta_G) = (
            dvar_sigma_rho_G, dvar_sigma_lambda_G, dvar_sigma_beta_G)
        (self.dvRE_rho, self.dvRE_lambda, self.dvRE_beta) = (
            dvRE_rho, dvRE_lambda, dvRE_beta)
        (self.d_rho_input, self.d_lambda_input, self.d_beta_input) = (
            d_rho_input, d_lambda_input, d_beta_input)

        # GARCH:

        self.dvar_omega = self.panel.arma_dot.dot(ll.GAR_1, panel.W_a, ll)
        self.dvar_initvar = None
        if 'initvar' in ll.args.args_d:
            self.dvar_initvar = ll.GAR_1[0].reshape((1, panel.max_T, 1))
            if not panel.options.EGARCH.value and ll.args.args_d['initvar'][0] < 0:
                self.dvar_initvar = -self.dvar_initvar

        (dvar_gamma, dvar_psi, dvar_mu, dvar_z_G,
         dvar_z) = (None, None, None, None, None)

        if panel.N > 1:
            dvar_mu = cf.prod((ll.dvarRE_mu, incl))
        else:
            dvar_mu = None

        if m > 0:
            dvar_gamma = self.arima_grad(k, var, ll, 1, ll.GAR_1)
            dvar_psi = self.arima_grad(m, h_val, ll, 1, ll.GAR_1)
            if not ll.h_z_val is None:
                dvar_z_G = fu.dot(ll.GAR_1MA, ll.h_z_val)
                (N, T, k) = dvar_z_G.shape

            dvar_z = dvar_z_G

        (self.dvar_gamma, self.dvar_psi, self.dvar_mu, self.dvar_z_G,
         self.dvar_z) = (dvar_gamma, dvar_psi, dvar_mu, dvar_z_G, dvar_z)

        # LL

        # final derivatives:
        dLL_beta = cf.add((cf.prod((dvar_sigma_beta, dLL_var)),
                          cf.prod((self.de_beta_RE, DLL_e))), True)
        dLL_rho = cf.add((cf.prod((dvar_sigma_rho, dLL_var)),
                         cf.prod((self.de_rho_RE, DLL_e))), True)
        dLL_lambda = cf.add((cf.prod((dvar_sigma_lambda, dLL_var)), cf.prod(
            (self.de_lambda_RE, DLL_e))), True)
        dLL_gamma = cf.prod((dvar_gamma, dLL_var))
        dLL_psi = cf.prod((dvar_psi, dLL_var))
        dLL_omega = cf.prod((self.dvar_omega, dLL_var))
        dLL_initvar = cf.prod((self.dvar_initvar, dLL_var))
        dLL_mu = cf.prod((self.dvar_mu, dLL_var))
        dLL_z = cf.prod((self.dvar_z, dLL_var))

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
        self.callback(perc=0.08, text='', task='gradient')

        if return_G:
            return g, G
        else:
            return g


class hessian:
    def __init__(self, panel, g, callback):
        self.panel = panel
        self.its = 0
        self.g = g
        self.callback = callback

    def get(self, ll, d2LL_de2, d2LL_dln_de, d2LL_dln2):
        H = self.hessian(ll, d2LL_de2, d2LL_dln_de, d2LL_dln2)
        return H

    def hessian(self, ll, d2LL_de2, d2LL_dln_de, d2LL_dln2):
        panel = self.panel
        tic = time.perf_counter()
        g = self.g
        p, q, d, k, m = panel.pqdkm
        incl = self.panel.included[3]

        GARM = (ll.GAR_1, m, 1)

        GARK = (ll.GAR_1, k, 1)

        d2var_gamma2 = cf.prod((2,
                                cf.dd_func_lags(panel, ll, GARK, 	g.dvar_gamma,						g.dLL_var,  transpose=True)))
        d2var_gamma_psi = cf.dd_func_lags(
            panel, ll, GARK,  g.dvar_psi,       g.dLL_var)
        d2var_gamma_rho = cf.dd_func_lags(
            panel, ll, GARK, g.dvar_sigma_rho_G,   g.dLL_var)
        d2var_gamma_lambda = cf.dd_func_lags(
            panel, ll, GARK,  g.dvar_sigma_lambda_G, g.dLL_var)
        d2var_gamma_beta = cf.dd_func_lags(
            panel, ll, GARK,  g.dvar_sigma_beta_G,  g.dLL_var)
        d2var_gamma_initvar = cf.dd_func_lags(
            panel, ll, GARK, 	g.dvar_initvar,					g.dLL_var)
        d2var_gamma_omega = cf.dd_func_lags(
            panel, ll, GARK, 	g.dvar_omega,					g.dLL_var)
        d2var_gamma_z = cf.dd_func_lags(
            panel, ll, GARK,  g.dvar_z_G,       g.dLL_var)

        self.callback(perc=0.2, text='', task='hessian')

        d2var_psi_rho = cf.dd_func_lags(panel, ll, GARM,  cf.prod(
            (ll.h_e_val, g.de_rho_RE)),  g.dLL_var)
        d2var_psi_lambda = cf.dd_func_lags(
            panel, ll, GARM,  cf.prod((ll.h_e_val, g.de_lambda_RE)), g.dLL_var)
        d2var_psi_beta = cf.dd_func_lags(
            panel, ll, GARM,  cf.prod((ll.h_e_val, g.de_beta_RE)), g.dLL_var)
        d2var_psi_z = cf.dd_func_lags(
            panel, ll, GARM,  ll.h_z_val,        g.dLL_var)

        AMAq = (ll.AMA_1, q, -1)
        d2var_lambda2,		d2e_lambda2 = cf.dd_func_lags_mult(
            panel, ll, g, AMAq, 'lambda', 'lambda', transpose=True)
        d2var_lambda_rho,	d2e_lambda_rho = cf.dd_func_lags_mult(
            panel, ll, g, AMAq, 'lambda', 'rho')
        d2var_lambda_beta,	d2e_lambda_beta = cf.dd_func_lags_mult(
            panel, ll, g, AMAq, 'lambda', 'beta')

        AMAp = (ll.AMA_1, p, -1)
        d2var_rho_beta,		d2e_rho_beta = cf.dd_func_lags_mult(
            panel, ll, g, AMAp, 'rho',  'beta', u_gradient=True)

        self.callback(perc=0.4, text='', task='hessian')

        d2var_mu_rho, d2var_mu_lambda, d2var_mu_beta, d2var_mu_z, mu = None, None, None, None, None
        if panel.N > 1:
            d2var_mu_rho = cf.sumNT(
                cf.prod((ll.ddvarRE_mu_vRE,  g.dvRE_rho,     g.dLL_var)))
            d2var_mu_lambda = cf.sumNT(
                cf.prod((ll.ddvarRE_mu_vRE,  g.dvRE_lambda,   g.dLL_var)))
            d2var_mu_beta = cf.sumNT(
                cf.prod((ll.ddvarRE_mu_vRE,  g.dvRE_beta,     g.dLL_var)))
            d2var_mu_z = None
            d2var_mu2 = 0

        self.callback(perc=0.5, text='', task='hessian')
        d2var_z2 = cf.dd_func_lags(
            panel, ll, ll.GAR_1MA, ll.h_2z_val,      g.dLL_var)
        d2var_z_rho = cf.dd_func_lags(panel, ll, ll.GAR_1MA, cf.prod(
            (ll.h_ez_val, g.de_rho_RE)), g.dLL_var)
        d2var_z_lambda = cf.dd_func_lags(panel, ll, ll.GAR_1MA, cf.prod(
            (ll.h_ez_val, g.de_lambda_RE)), g.dLL_var)
        d2var_z_beta = cf.dd_func_lags(panel, ll, ll.GAR_1MA, cf.prod(
            (ll.h_ez_val, g.de_beta_RE)), g.dLL_var)

        d2var_rho2,	d2e_rho2 = cf.dd_func_lags_mult(
            panel, ll, g, None, 'rho',  'rho')
        d2var_beta2, d2e_beta2 = cf.dd_func_lags_mult(
            panel, ll, g, None, 'beta',  'beta')

        (de_rho_RE, de_lambda_RE, de_beta_RE) = (
            g.de_rho_RE, g.de_lambda_RE, g.de_beta_RE)
        (dvar_sigma_rho, dvar_sigma_lambda, dvar_sigma_beta) = (
            g.dvar_sigma_rho, g.dvar_sigma_lambda, g.dvar_sigma_beta)
        (dvar_mu, dvar_z) = (g.dvar_mu, g.dvar_z)

        d2var_beta_omega, d2var_beta_initvar, d2var_rho_omega, d2var_lambda_omega = None, None, None, None
        d2var_rho_initvar, d2var_lambda_initvar = None, None

        self.callback(perc=0.6, text='', task='hessian')
        # Final:
        D2LL_beta2 = cf.dd_func(d2LL_de2, d2LL_dln_de, d2LL_dln2, de_beta_RE,  de_beta_RE,
                                dvar_sigma_beta,  dvar_sigma_beta, d2e_beta2,      d2var_beta2)
        D2LL_beta_rho = cf.dd_func(d2LL_de2, d2LL_dln_de, d2LL_dln2, de_beta_RE,  de_rho_RE,
                                   dvar_sigma_beta,  dvar_sigma_rho,  T(d2e_rho_beta),   T(d2var_rho_beta))
        D2LL_beta_lambda = cf.dd_func(d2LL_de2, d2LL_dln_de, d2LL_dln2, de_beta_RE,  de_lambda_RE,
                                      dvar_sigma_beta,  dvar_sigma_lambda, T(d2e_lambda_beta),  T(d2var_lambda_beta))
        D2LL_beta_gamma = cf.dd_func(None,  d2LL_dln_de, d2LL_dln2, de_beta_RE,
                                     None,   dvar_sigma_beta,  g.dvar_gamma,  None,      T(d2var_gamma_beta))
        D2LL_beta_psi = cf.dd_func(None,  d2LL_dln_de, d2LL_dln2, de_beta_RE,
                                   None,   dvar_sigma_beta,  g.dvar_psi,   None,      T(d2var_psi_beta))
        D2LL_beta_omega = cf.dd_func(None,  d2LL_dln_de, d2LL_dln2, de_beta_RE,
                                     None,   dvar_sigma_beta,  g.dvar_omega,  None,      d2var_beta_omega)
        D2LL_beta_initvar = cf.dd_func(None,  d2LL_dln_de, d2LL_dln2, de_beta_RE,
                                       None,   dvar_sigma_beta,  g.dvar_initvar, None,      d2var_beta_initvar)
        D2LL_beta_mu = cf.dd_func(None,  d2LL_dln_de, d2LL_dln2, de_beta_RE,
                                  None,   dvar_sigma_beta,  dvar_mu,    None,      d2var_mu_beta)
        D2LL_beta_z = cf.dd_func(None,  d2LL_dln_de, d2LL_dln2, de_beta_RE,
                                 None,   dvar_sigma_beta,  dvar_z,     None,      T(d2var_z_beta))

        D2LL_rho2 = cf.dd_func(d2LL_de2, d2LL_dln_de, d2LL_dln2, de_rho_RE,
                               de_rho_RE,  dvar_sigma_rho,  dvar_sigma_rho,  d2e_rho2,      d2var_rho2)
        D2LL_rho_lambda = cf.dd_func(d2LL_de2, d2LL_dln_de, d2LL_dln2, de_rho_RE,   de_lambda_RE,
                                     dvar_sigma_rho,  dvar_sigma_lambda, T(d2e_lambda_rho),   T(d2var_lambda_rho))
        D2LL_rho_gamma = cf.dd_func(None,  d2LL_dln_de, d2LL_dln2, de_rho_RE,
                                    None,   dvar_sigma_rho,  g.dvar_gamma,  None,      T(d2var_gamma_rho))
        D2LL_rho_psi = cf.dd_func(None,  d2LL_dln_de, d2LL_dln2, de_rho_RE,
                                  None,   dvar_sigma_rho,  g.dvar_psi,   None,      T(d2var_psi_rho))
        D2LL_rho_omega = cf.dd_func(None,  d2LL_dln_de, d2LL_dln2, de_rho_RE,
                                    None,   dvar_sigma_rho,  g.dvar_omega,  None,      d2var_rho_omega)
        D2LL_rho_initvar = cf.dd_func(None,  d2LL_dln_de, d2LL_dln2, de_rho_RE,
                                      None,   dvar_sigma_rho,  g.dvar_initvar, None,      d2var_rho_initvar)
        D2LL_rho_mu = cf.dd_func(None,  d2LL_dln_de, d2LL_dln2, de_rho_RE,
                                 None,   dvar_sigma_rho,  dvar_mu,    None,      T(d2var_mu_rho))
        D2LL_rho_z = cf.dd_func(None,  d2LL_dln_de, d2LL_dln2, de_rho_RE,
                                None,   dvar_sigma_rho,  dvar_z,     None,      T(d2var_z_rho))

        D2LL_lambda2 = cf.dd_func(d2LL_de2, d2LL_dln_de, d2LL_dln2, de_lambda_RE,  de_lambda_RE,
                                  dvar_sigma_lambda,  dvar_sigma_lambda, T(d2e_lambda2),   T(d2var_lambda2))
        D2LL_lambda_gamma = cf.dd_func(None,  d2LL_dln_de, d2LL_dln2, de_lambda_RE,
                                       None,   dvar_sigma_lambda,  g.dvar_gamma,  None,      T(d2var_gamma_lambda))
        D2LL_lambda_psi = cf.dd_func(None,  d2LL_dln_de, d2LL_dln2, de_lambda_RE,
                                     None,   dvar_sigma_lambda,  g.dvar_psi,   None,      T(d2var_psi_lambda))
        D2LL_lambda_omega = cf.dd_func(None,  d2LL_dln_de, d2LL_dln2, de_lambda_RE,
                                       None,   dvar_sigma_lambda,  g.dvar_omega,  None,      d2var_lambda_omega)
        D2LL_lambda_initvar = cf.dd_func(None,  d2LL_dln_de, d2LL_dln2, de_lambda_RE,
                                         None,   dvar_sigma_lambda,  g.dvar_initvar, None,      d2var_lambda_initvar)
        D2LL_lambda_mu = cf.dd_func(None,  d2LL_dln_de, d2LL_dln2, de_lambda_RE,
                                    None,   dvar_sigma_lambda,  dvar_mu,    None,      T(d2var_mu_lambda))
        D2LL_lambda_z = cf.dd_func(None,  d2LL_dln_de, d2LL_dln2, de_lambda_RE,
                                   None,   dvar_sigma_lambda,  dvar_z,     None,      T(d2var_z_lambda))

        D2LL_gamma2 = cf.dd_func(None,  None,   d2LL_dln2, None,   None,
                                 g.dvar_gamma,   g.dvar_gamma,  None,      T(d2var_gamma2))
        D2LL_gamma_psi = cf.dd_func(None,  None,   d2LL_dln2, None,   None,
                                    g.dvar_gamma,   g.dvar_psi,   None,      d2var_gamma_psi)
        D2LL_gamma_omega = cf.dd_func(None,  None,   d2LL_dln2, None,   None,
                                      g.dvar_gamma,   g.dvar_omega,  None,      d2var_gamma_omega)
        D2LL_gamma_initvar = cf.dd_func(None,  None,   d2LL_dln2, None,   None,
                                        g.dvar_gamma,    g.dvar_initvar, None,      d2var_gamma_initvar)
        D2LL_gamma_mu = cf.dd_func(
            None,  None,   d2LL_dln2, None,   None,   g.dvar_gamma,   dvar_mu,    None,      None)
        D2LL_gamma_z = cf.dd_func(None,  None,   d2LL_dln2, None,
                                  None,   g.dvar_gamma,   dvar_z,     None,      d2var_gamma_z)

        D2LL_psi2 = cf.dd_func(None,  None,   d2LL_dln2, None,
                               None,   g.dvar_psi,   g.dvar_psi,   None,      None)
        D2LL_psi_omega = cf.dd_func(
            None,  None,   d2LL_dln2, None,   None,   g.dvar_psi,   g.dvar_omega,  None,      None)
        D2LL_psi_initvar = cf.dd_func(
            None,  None,   d2LL_dln2, None,   None,   g.dvar_psi,    g.dvar_initvar, None,      None)
        D2LL_psi_mu = cf.dd_func(None,  None,   d2LL_dln2, None,
                                 None,   g.dvar_psi,   dvar_mu,     None,      None)
        D2LL_psi_z = cf.dd_func(None,  None,   d2LL_dln2, None,
                                None,   g.dvar_psi,   dvar_z,      None,      d2var_psi_z)

        D2LL_omega2 = cf.dd_func(None,  None,   d2LL_dln2, None,
                                 None,   g.dvar_omega,  g.dvar_omega,  None,      None)
        D2LL_omega_initvar = cf.dd_func(
            None,  None,     d2LL_dln2, None,      None,   g.dvar_omega,  g.dvar_initvar, None,      None)
        D2LL_omega_mu = cf.dd_func(
            None,  None,   d2LL_dln2, None,    None,   g.dvar_omega,  g.dvar_mu,   None,      None)
        D2LL_omega_z = cf.dd_func(None,  None,   d2LL_dln2, None,
                                  None,   g.dvar_omega,  g.dvar_z,     None,      None)

        D2LL_initvar2 = cf.dd_func(None,  None,   d2LL_dln2, None,
                                   None,   g.dvar_initvar,  g.dvar_initvar,  None,      None)
        D2LL_initvar_mu = cf.dd_func(
            None,  None,   d2LL_dln2, None,    None,   dvar_mu,      g.dvar_initvar,  None,      None)
        D2LL_initvar_z = cf.dd_func(
            None,  None,   d2LL_dln2, None,    None,   dvar_z,      g.dvar_initvar,  None,      None)

        D2LL_mu2 = cf.dd_func(None,  None,   d2LL_dln2, None,
                              None,   dvar_mu,    dvar_mu,   None,      None)
        D2LL_mu_z = cf.dd_func(None,  None,   d2LL_dln2, None,
                               None,   dvar_mu,    dvar_z,    None,      d2var_mu_z)

        D2LL_z2 = cf.dd_func(None,  None,   d2LL_dln2, None,
                             None,   dvar_z,    dvar_z,    None,      d2var_z2)

        H = [[D2LL_beta2,					D2LL_beta_rho,				D2LL_beta_lambda,				D2LL_beta_gamma,			D2LL_beta_psi,			D2LL_beta_omega,				D2LL_beta_initvar, 		D2LL_beta_mu,		D2LL_beta_z],
             [T(D2LL_beta_rho),		D2LL_rho2,						D2LL_rho_lambda,				D2LL_rho_gamma,
              D2LL_rho_psi,				D2LL_rho_omega,					D2LL_rho_initvar, 		D2LL_rho_mu,		D2LL_rho_z],
             [T(D2LL_beta_lambda),	T(D2LL_rho_lambda),		D2LL_lambda2,						D2LL_lambda_gamma,
              D2LL_lambda_psi,		D2LL_lambda_omega,			D2LL_lambda_initvar, 	D2LL_lambda_mu,	D2LL_lambda_z],
             [T(D2LL_beta_gamma),	T(D2LL_rho_gamma),		T(D2LL_lambda_gamma),		D2LL_gamma2,
              D2LL_gamma_psi,			D2LL_gamma_omega, 			D2LL_gamma_initvar, 	D2LL_gamma_mu,	D2LL_gamma_z],
             [T(D2LL_beta_psi),		T(D2LL_rho_psi),			T(D2LL_lambda_psi),			T(D2LL_gamma_psi),
              D2LL_psi2,					D2LL_psi_omega, 				D2LL_psi_initvar, 		D2LL_psi_mu,		D2LL_psi_z],
             [T(D2LL_beta_omega),	T(D2LL_rho_omega),		T(D2LL_lambda_omega),		T(D2LL_gamma_omega),	T(
                 D2LL_psi_omega),	D2LL_omega2, 						D2LL_omega_initvar, 	D2LL_omega_mu,	D2LL_omega_z],
             [T(D2LL_beta_initvar), T(D2LL_rho_initvar),	T(D2LL_lambda_initvar),	T(D2LL_gamma_initvar), T(
                 D2LL_psi_initvar), T(D2LL_omega_initvar), 	D2LL_initvar2, 				D2LL_initvar_mu, D2LL_initvar_z],
             [T(D2LL_beta_mu),			T(D2LL_rho_mu),				T(D2LL_lambda_mu),			T(D2LL_gamma_mu),			T(
                 D2LL_psi_mu),			T(D2LL_omega_mu), 			D2LL_initvar_mu, 			D2LL_mu2,				D2LL_mu_z],
             [T(D2LL_beta_z),			T(D2LL_rho_z),				T(D2LL_lambda_z),				T(D2LL_gamma_z),			T(D2LL_psi_z),			T(D2LL_omega_z), 				D2LL_initvar_z, 			D2LL_mu_z,			D2LL_z2]]
        self.callback(perc=0.8, text='', task='hessian')
        H = cf.concat_matrix(H)
        if H[-1, -1] == 0:
            H[-1, -1] = 1
        # for debugging:
        # from .. import debug
        # Hn=debug.hess_debug(ll,panel,g,0.00000001)#debugging
        # v=debug.hess_debug_detail(ll,panel,0.0000001,'grp','beta','beta',0,0)
        # print (time.perf_counter()-tic)
        self.its += 1
        if np.any(np.isnan(H)):
            return None
        # print(H[0]/1e+11)
        return H


def T(x):
    if x is None:
        return None
    return x.T
