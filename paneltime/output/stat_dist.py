#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Generates numerical distribution functions that don't require the slow loading of scipy

import mpmath as mp
import numpy as np
mp.mp.dps = 50


def tcdf(x, df):
    "Cumulative distribution function of the t-distribution"
    if df < 1:
        return 0
    try:
        len(x)
    except:
        dist = 0.5*(1 + mp.betainc(df/2, 0.5, df /
                    (df + abs(x) ** 2), regularized=True))
        if x < 0:
            dist = 1-dist
        return float(dist)
    if len(x) > 100:
        raise RuntimeWarning(
            f"Dimension is {len(x)}. This function is not made for such large dimension. Consider using scipy. ")
    return np.array([tcdf(i, df) for i in x])


def tinv025(df):
    z = [0, 12.70620474, 4.30265273, 3.182446305, 2.776445105, 2.570581836, 2.446911851, 2.364624252, 2.306004135,
         2.262157163, 2.228138852, 2.20098516, 2.17881283, 2.160368656, 2.144786688, 2.131449546, 2.119905299,
         2.109815578, 2.10092204, 2.093024054, 2.085963447, 2.079613845, 2.073873068, 2.06865761, 2.063898562,
         2.059538553, 2.055529439, 2.051830516, 2.048407142, 2.045229642, 2.042272456, 2.039513446, 2.036933343,
         2.034515297, 2.032244509, 2.030107928, 2.028094001, 2.026192463, 2.024394164, 2.02269092, 2.02107539,
         2.01954097, 2.018081703, 2.016692199, 2.015367574, 2.014103389, 2.012895599, 2.011740514, 2.010634758,
         2.009575237, 2.008559112]
    if df < len(z):
        return z[df]
    t = (z[2] - 1.959963985) / (df - 1) + 1.959963985
    return t


def norm(x, mu=0, s=1, cdf=True):
    try:
        len(x)
    except:
        if not cdf:
            return mp.npdf(x, mu, s)
        f = 0.5*mp.erfc((mu-x)/(s*2**0.5))
        return float(f)
    if len(x) > 100:
        raise RuntimeWarning(
            f"Dimension is {len(x)}. This function is not made for such large dimension. Consider using scipy. ")
    return np.array([norm(i, mu, s, cdf) for i in x])


def chisq(x, k):
    "cdf of chi-square distribution of x for k degrees of freedom"
    if x <= 0 or k <= 0:
        return 0
    c = mp.gammainc(0.5*k, 0, 0.5*x)/mp.gamma(0.5*k)
    if mp.im(c) != 0:
        return 0
    return float(c)


def fcdf(x, k1, k2):
    f = mp.betainc(0.5*k1, 0.5*k2, x1=0, x2=k1*x/(k1*x+k2), regularized=True)
    if mp.im(f) != 0:
        return None
    return float(f)


def test_functions():
    from scipy import stats as scstats
    print(tcdf(3, 5))
    print(scstats.t.cdf(3, 5))

    print(norm(3, 5, 2))
    print(scstats.norm.cdf(3, 5, 2))

    print(chisq(3, 5))
    print(scstats.chi2.cdf(3, 5))

    print(1-fcdf(111, 4, 7228))
    print(1-scstats.f.cdf(111, 4, 7228))

    print(norm(3, 5, 1, False))
    print(scstats.norm.pdf(3, 5, 1))

    print(tinv025(100))
    print(scstats.t.ppf(0.025, 100))
    a = 0


# test_functions()
a = 0
