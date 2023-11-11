#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import dfpmax
from . import computation
# from ..likelihood_simple import dfpmax as dfpmax_smpl
from .. import system_settings
if system_settings.cython:
    from .. import likelihood_cython as logl
else:
    from .. import likelihood as logl

import numpy as np
from ..parallel import callback


def maximize(args, inbox, outbox, panel, gtol, tolx, nummerical, diag_hess, slave_id):
    args = np.array(args)
    callbk = callback.CallBack(inbox, outbox)
    comput = computation.Computation(
        args, panel, gtol, tolx, None, nummerical, diag_hess)

    callbk.callback(quit=False, conv=False, perc=0)
    initval = InitialValues(panel, comput)

    x, ll, f, g, hessin, H = initval.calc_init_dir(args, panel)
    res, ll = dfpmax.dfpmax(x, f, g, hessin, H, comput,
                            callbk, panel, slave_id, ll)
    # msg, conv, args_c = dfpmax_smpl.dfpmax(x, f, g, hessin, H, panel, slave_id)

    return res, ll


class InitialValues:
    def __init__(self, panel, comput):
        self.comput = comput
        self.panel = panel

    def init_ll(self, args=None, ll=None):
        if args is None:
            args = ll.args.args_v

        try:
            args = args.args_v
        except:
            pass  # args must be a vector
        for i in self.comput.constr.fixed:
            args[i] = self.comput.constr.fixed[i].value
        if ll is None:
            ll = logl.LL(args, self.panel,
                         constraints=self.comput.constr, print_err=True)
        if ll.LL is None:
            print(
                "WARNING: Initial arguments failed, attempting default OLS-arguments ...")
            self.panel.args.set_init_args(self.panel, default=True)
            ll = logl.LL(self.panel.args.args_OLS, self.panel,
                         constraints=self.constr, print_err=True)
            if ll.LL is None:
                raise RuntimeError(
                    "OLS-arguments failed too, you should check the data")
            else:
                print("default OLS-arguments worked")
        return ll

    def calc_init_dir(self, p0, panel, diag_hess=False):
        """Calculates the initial computation"""
        ll = self.init_ll(p0)
        g, G = self.comput.calc_gradient(ll)
        if self.panel.options.use_analytical.value == 2:
            H = -np.identity(len(g))
            hessin = H
            return p0, ll, ll.LL, g, hessin, H
        H = self.comput.calc_hessian(ll)
        if diag_hess:
            d = np.diag(H)
            d = d - (np.abs(d) < 1e-100)
            H = np.diag(d)
            hessin = np.diag(1/d)
        else:
            try:
                hessin = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                hessin = -np.identity(len(g))*panel.args.init_var
        return p0, ll, ll.LL, g, hessin, H
