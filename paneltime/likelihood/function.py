#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
MIN_DEN = 0


def LL(panel, var, e_REsq, e_RE, minvar, maxvar):
    incl = panel.included[3]

    LL_const = -0.5*np.log(2*np.pi)
    if panel.options.EGARCH.value == 0:
        a, k = panel.options.GARCH_assist.value, panel.options.kurtosis_adj.value

        dvar_pos = (var < maxvar)*(var > minvar)
        var = incl*np.maximum(np.minimum(var, maxvar), minvar)
        v = var
        v_inv = incl/(var + MIN_DEN + (incl == 0))

        LL_full = LL_const-0.5*(incl*np.log(var+MIN_DEN + (incl == 0))+(1-k)*e_REsq*v_inv
                                + a * (np.abs(e_REsq-var)*v_inv)
                                + (k/3) * e_REsq**2*v_inv**2
                                )
    else:
        dvar_pos = (var < maxvar) * (var > minvar)
        var = np.maximum(np.minimum(var, maxvar), minvar)
        v = np.exp(var)*incl
        v_inv = np.exp(-var)*incl
        LL_full = LL_const-0.5*(var+(e_REsq)*v_inv)
    return LL_full, v, v_inv, dvar_pos


def gradient(ll, panel):
    incl = panel.included[3]
    a, k = panel.options.GARCH_assist.value, panel.options.kurtosis_adj.value
    var, e_REsq, e_RE, v_inv = ll.var, ll.e_REsq, ll.e_RE, ll.v_inv

    if panel.options.EGARCH.value == 0:
        DLL_e = -0.5*((1-k)*2*e_RE*v_inv)
        dLL_var = -0.5*(v_inv-(1-k)*(e_REsq)*v_inv**2)

        DLL_e += -0.5*(
            a * 2*np.sign(e_REsq-var)*e_RE*v_inv
            + (k/3) * 4*e_REsq*e_RE*v_inv**2
        )
        dLL_var += -0.5*(
            - a * (np.sign(e_REsq-var)*v_inv)
            - a * (np.abs(e_REsq-var)*v_inv**2)
            - (k/3) * 2*e_REsq**2*v_inv**3
        )
    else:
        DLL_e = -(ll.e_RE*ll.v_inv)
        dLL_var = -0.5*(incl-(ll.e_REsq*ll.v_inv)*incl)
    dLL_var *= ll.dvar_pos*incl
    DLL_e *= incl
    return dLL_var, DLL_e


def hessian(ll, panel):
    incl = panel.included[3]
    var, e_REsq, e_RE, v_inv = ll.var, ll.e_REsq, ll.e_RE, ll.v_inv
    a, k = panel.options.GARCH_assist.value, panel.options.kurtosis_adj.value

    if panel.options.EGARCH.value == 0:
        d2LL_de2 = -0.5*((1-k)*2*v_inv)
        d2LL_dln_de = -0.5*(-(1-k)*2*e_RE*v_inv**2)
        d2LL_dln2 = -0.5*(-1*v_inv**2+(1-k)*2*(e_REsq)*v_inv**3)

        d2LL_de2 += -0.5*(
            a * 2*np.sign(e_REsq-var)*v_inv
            + (k/3) * 12*e_REsq*v_inv**2
        )
        d2LL_dln_de += -0.5*(
            - a * 2*np.sign(e_REsq-var)*e_RE*v_inv**2
            - (k/3) * 8*e_REsq*e_RE*v_inv**3
        )
        d2LL_dln2 += -0.5*(
            + a * (np.sign(e_REsq-var)*v_inv**2)
            + a * 2*(np.abs(e_REsq-var)*v_inv**3)
            + a * (np.sign(e_REsq-var)*v_inv**2)
            + (k/3) * 6 *
            e_REsq**2*v_inv**4
        )

    else:
        d2LL_de2 = -ll.v_inv
        d2LL_dln_de = ll.e_RE*ll.v_inv
        d2LL_dln2 = -0.5*ll.e_REsq*ll.v_inv
    d2LL_dln_de *= ll.dvar_pos*incl
    d2LL_dln2 *= ll.dvar_pos*incl
    d2LL_de2 *= incl

    return d2LL_de2, d2LL_dln_de, d2LL_dln2
