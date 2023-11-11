#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import linesearch
from . import direction
from . import constraints
from .. import likelihood_simple as logl
import numpy as np
import time

# This module finds the array of arguments that minimizes some function. The derivative
# of the function also needs to be supplied.
# This is an adaption of the Broyden-Fletcher-Goldfarb-Shanno variant of Davidon-Fletcher-Powell algorithm by
# Press, William H,Saul A Teukolsky,William T Vetterling and Brian P Flannery. 1992. Numerical Recipes in C'.
# Cambridge: Cambridge University Press.


EPS = 3.0e-16
TOLX = (4*EPS)
GTOL = 0.001


def dfpmax(x, f, g, hessin, H, panel, slave_id):
    """Given a starting point x[1..n] that is a vector of length n, the Broyden-Fletcher-Goldfarb-
    Shanno variant of Davidon-Fletcher-Powell minimization is performed on a function func, using
    its gradient as calculated by a routine dfunc. The convergence requirement on zeroing the
    gradient is input as gtol. Returned quantities are x[1..n] (the location of the minimum),
    iter (the number of iterations that were performed), and fret (the minimum value of the
    function). The routine lnsrch is called to perform approximate line minimizations.
    fargs are fixed arguments that ar not subject to optimization. ("Nummerical Recipes for C") """

    ll = logl.LL(x, panel)

    its, msg = 0, ''
    MAXITER = 10000

    for its in range(MAXITER):  # Main loop over the iterations.
        constr = get_constr(its, H, ll, x, panel)
        dx, dx_norm, H_ = direction.get(
            g, x, H, constr, f, hessin, simple=False)
        ls = linesearch.LineSearch(x, panel, ll)
        ls.lnsrch(x, f, g, H, dx, constr)

        dx_realized = ls.x - x
        incr = ls.f - f

        ll = ls.ll

        x, f, hessin, H, g, conv = calc(
            dx_realized,  hessin, H, incr, its, ls, panel, constr)

        err = (np.max(np.abs(dx_realized)) < TOLX) and its > 5

        if conv == 1:
            msg = "Convergence on zero gradient; local or global minimum identified"
        elif conv == 2:
            msg = "Convergence on zero expected gain; local or global minimum identified given multicolinearity constraints"
        elif conv == 3:
            msg = "Reached the maximum number of iterations"
        elif err:
            msg = "Warning: Convergence on delta x; the gradient is incorrect or the tolerance is set too low"
        elif its > MAXITER:
            msg = "No convergence within %s iterations" % (MAXITER,)

        if (conv > 0) or err or its+1 == MAXITER:
            return msg, conv, ls.ll.args.args_v


def get_constr(its, H, ll, x, panel):
    constr = constraints.Constraints(panel, x)
    constr.add_static_constraints(
        panel, its, ll,  np.nonzero(ll.var < ll.minvar)[1])
    constr.add_dynamic_constraints(panel, H, ll)
    return constr


def calc(dx_realized, hessin, H, incr, its, ls, panel, constr):
    f, x, g_old, rev, alam, ll = ls.f, ls.x, ls.g, ls.rev, ls.alam, ls.ll
    # Thhese setting may not hold for all circumstances, and should be tested properly:

    g, G = calc_gradient(ll, panel)

    a = np.ones(len(g))
    if not constr is None:
        a[list(constr.fixed.keys())] = 0
        a[ls.applied_constraints] = 0
    g_norm = np.max(np.abs(g*a*x)/(abs(f)+1e-12))
    gtol = GTOL
    if sum(a) == 1:
        gtol = 1e-10

    if abs(g_norm) < gtol:
        return x, f, hessin, H, g, 1

    print(f"its:{its}, f:{f}, gnorm: {abs(g_norm)}")

    hessin = hessin_num(hessin, g-g_old, dx_realized)
    H = inv(hessin, H)

    return x, f, hessin, H, g, 0


def calc_gradient(ll, panel):
    dLL_lnv, DLL_e = logl.func_gradent(ll, panel)
    g, G = logl.gradient(ll, panel, DLL_e, dLL_lnv, return_G=True)
    return g, G


def hessin_num(hessin, dg, xi):  # Compute difference of gradients,
    n = len(dg)
    # and difference times current matrix:
    hdg = (np.dot(hessin, dg.reshape(n, 1))).flatten()
    # Calculate dot products for the denominators.
    fac = fae = sumdg = sumxi = 0.0
    fac = np.sum(dg*xi)
    fae = np.sum(dg*hdg)
    sumdg = np.sum(dg*dg)
    sumxi = np.sum(xi*xi)
    # Skip update if fac not sufficiently positive.
    if (fac < (EPS*sumdg*sumxi)**0.5):
        fac = 1.0/fac
        fad = 1.0/fae
        # The vector that makes BFGS different from DFP:
        dg = fac*xi-fad*hdg
        # The BFGS updating formula:
        hessin += fac*xi.reshape(n, 1)*xi.reshape(1, n)
        hessin -= fad*hdg.reshape(n, 1)*hdg.reshape(1, n)
        hessin += fae*dg.reshape(n, 1)*dg.reshape(1, n)

    return hessin


def inv(hessian, H):
    try:
        return np.linalg.inv(hessian)
    except:
        return H
