#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from .. import likelihood_simple as logl
from . import direction
import numpy as np

STPMX = 100.0


class LineSearch:
    def __init__(self, x, panel, ll_old, step=1):
        self.alf = 1.0e-3  # Ensures sufficient decrease in function value.
        self.tolx = 1.0e-14  # Convergence criterion on fx.
        self.step = step
        self.stpmax = STPMX * max((abs(np.sum(x**2)))**0.5, len(x))
        self.panel = panel
        self.applied_constraints = []
        self.ll_old = ll_old

    def lnsrch(self, x, f, g, H, dx, constr):

        # (x, f, g, dx)

        self.conv = 0
        self.rev = False
        self.g = g
        self.msg = ""
        n = len(x)

        if f is None:
            raise RuntimeError('f cannot be None')

        summ = np.sum(dx**2)**0.5
        if summ > self.stpmax:
            dx = dx*self.stpmax/summ
        if np.all(dx == 0):
            self.default(f, x, 0,  "dx is zero", 4)
            return

        test = 0.0  # Compute lambda min.
        for i in range(0, n):
            temp = abs(dx[i])/max(abs(x[i]), 1.0)
            if (temp > test):
                test = temp
        alamin = self.tolx/test
        # *******CUSTOMIZATION
        # multithread:

        # Setting alam so that the largest step is valid. Set func to return None when input is invalid
        for i in range(1000):
            self.alam = 0.5**i*self.step  # Always try full Newton step first.
            dx_alam, slope, self.rev, self.applied_constraints = direction.new(
                g, x, H, constr, f, dx, self.alam)
            self.x = x + dx_alam
            self.f, self.ll = self.func(self.x, constr)
            if self.f != None:
                break
        # *************************
        f2 = 0
        alam2 = self.alam
        alamstart = self.alam  # ***********CUSTOMIZATION
        max_iter = 1000
        for self.k in range(0, max_iter):  # Start of iteration loop.
            dx_alam, slope, self.rev, self.applied_constraints = direction.new(
                g, x, H, constr, f, dx, self.alam)
            self.x = x + dx_alam
            if self.k > 0:
                self.f, self.ll = self.func(self.x, constr)
            if self.f is None:
                self.msg = 'The function returned None'
                self.f = f
            # Convergence on delta x. For zero finding,the calling program should verify the convergence.
            if (self.alam < alamin):
                self.default(f, x, 0, "Convergence on delta dx", 2)
                return
            elif (self.f >= f+self.alf*self.alam*slope):
                self.msg = "Sufficient function increase"
                # print(f'LL single:{self.ll.LL}')
                # self.ll = self.ll2
                self.conv = 1
                return  # Sufficient function increase
            else:  # Backtrack.
                if (self.alam == alamstart):  # ***********CUSTOMIZATION  alam == 1.0
                    tmplam = -slope/(2.0*(self.f-f-slope))  # First time.
                else:  # Subsequent backtracks.
                    rhs1 = self.f-f-self.alam*slope
                    rhs2 = f2-f-alam2*slope
                    a = (rhs1/(self.alam**2)-rhs2 /
                         (alam2*alam2))/(self.alam-alam2)
                    b = (-alam2*rhs1/(self.alam**2)+self.alam *
                         rhs2/(alam2*alam2))/(self.alam-alam2)
                    if (a == 0.0):
                        tmplam = -slope/(2.0*b)
                    else:
                        disc = b*b-3.0*a*slope
                        if (disc < 0.0):
                            tmplam = 0.5*self.alam
                        elif (b >= 0.0):
                            tmplam = -(b+(disc)**0.5)/(3.0*a)
                        else:
                            tmplam = slope/(-b+(disc)**0.5)
                    if (tmplam > 0.5*self.alam):
                        tmplam = 0.5*self.alam  # lambda<=0.5*lambda1
            alam2 = self.alam
            f2 = self.f
            self.alam = max(tmplam, 0.1*self.alam)  # lambda>=0.1*lambda1
            if alamstart < 1.0:  # *************CUSTOMIZATION
                self.alam = min((self.alam, alamstart*0.9**self.k))

        self.msg = f"No function increase after {max_iter} iterations"
        self.conv = 3

    def func(self, x, constr):
        ll = logl.LL(x, self.panel, constr)
        if ll is None:
            return None, None
        elif ll.LL is None:
            return None, None
        return ll.LL, ll

    def default(self, f, x, alam, msg, conv):
        self.msg = msg
        self.conv = conv
        self.f = f
        self.x = x
        self.alam = alam
        self.ll = self.ll_old
