#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def save_csv(fname, array, sep=','):
    f = open(fname, 'wt')
    try:
        for line in array:
            f.write(sep.join(line))
    except TypeError as e:
        np.savetxt(fname, array, fmt='%s', delimiter=sep)
    f.close()


def dot(a, b, reduce_dims=True):
    """Matrix multiplication. Returns the dot product of a*b where either a or be or both to be
    arrays of matrices. Faster than mmult, less general and only used for special purpose.
    Todo: generalize and merge"""

    if len(a.shape) == 3 and len(b.shape) == 2:
        x = np.array([np.dot(a[i], b) for i in range(a.shape[0])])
    elif len(a.shape) == 3 and len(b.shape) == 3:
        x = np.sum([np.dot(a[i].T, b[i]) for i in range(a.shape[0])], 0)
    elif len(a.shape) == 2 and len(b.shape) == 3:
        x = np.array([np.dot(a, b[i]) for i in range(b.shape[0])])
    elif len(a.shape) == 2 and len(b.shape) == 2:
        if a.shape[1] == b.shape[0]:
            x = np.dot(a, b)
    return x


class ArmaDot:
    def __init__(self):
        pass

    def dotroll(self, aband, k, sign, b, ll):
        x = sign*self.fast_dot(aband, b)
        w = []
        for i in range(k):
            w.append(np.roll(np.array(x), i+1, 1))
            w[i][:, :i+1] = 0
        x = np.array(w)
        x = np.moveaxis(x, 0, 2)
        return x

    def fast_dot(self, a, b):
        a_, name = a
        n = get_n(a_)
        if n is None:
            n = len(a_)

        r = a_[0]*b
        for i in range(1, n):
            r[:, i:] += a_[i]*b[:, :-i]

        return r

    def dot(self, a, b, ll):
        if len(a) > 2:  # then this is a proper matrix
            (aband, k, sgn) = a
            if k == 0:
                return None
            return self.dotroll(aband, k, sgn, b, ll)
        x = self.fast_dot(a, b)
        return x


def get_n(a):
    minval = 0
    a_1 = np.abs(a[1:])
    max_a = np.max(a_1)
    if np.min(np.abs(a_1)) >= minval:
        return None
    if max_a == 0:
        return 1
    else:
        nz = np.nonzero(a_1/max_a < minval)[0]
        if len(nz) > 0:
            return nz[0]+1
        else:
            return None
