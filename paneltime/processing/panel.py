#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This module contains classes used in the regression
from .. import system_settings
if system_settings.cython:
    from .. import likelihood_cython as logl
else:
    from .. import likelihood as logl

from . import arguments
from .. import functions as fu

import numpy as np
import time


min_AC = 0.000001

# todo: split up panel in sub classes


def posdef(a, da):
    return list(range(a, a+da)), a+da


class panel:
    def __init__(self, datainput, settings, pqdkm=None):

        self.input = datainput
        self.options = settings
        self.dataframe = datainput.dataframe
        if not pqdkm is None:
            self.pqdkm = pqdkm
        else:
            self.pqdkm = settings.pqdkm.value

    def init(self):
        self.initial_defs()
        self.arrayize()
        self.masking()
        self.lag_variables()
        self.final_defs()
        self.arma_dot = fu.ArmaDot()

    def initial_defs(self):
        if np.all(np.var(self.input.Y, 0) == 0):
            raise RuntimeError("No variation in Y")
        p, q, d, k, m = self.pqdkm
        self.orig_size = len(self.input.X)
        self.max_lags = self.input.max_lags
        self.lost_obs = max((p, q, self.max_lags)) + \
            max((m, k, self.max_lags))+d  # +3
        self.nW, self.nZ, self.n_beta = len(self.input.W.columns), len(
            self.input.Z.columns), len(self.input.X.columns)
        self.define_h_func()
        self.Ld_inv = None
        if self.input.IDs is None:
            self.options.fixed_random_group_eff.value = 0
        if self.input.IDs is None:
            self.options.fixed_random_group_eff.value = 0
        self.m_zero = False
        if m == 0 and k > 0:
            self.m_zero = True
            k = 0
            print("Warning: GARCH term removed since ARCH terms was set to 0")
        self.pqdkm = p, q, d, k, m

    def masking(self):

        # "initial observations" mask:
        # sets observations that shall be zero to zero by multiplying it with the arrayized variable
        a = np.array([self.date_counter < self.T_arr[i]
                     for i in range(self.N)])
        self.a = [None, None]
        self.a.extend([a.reshape(list(a.shape)[:-1]+[1]*i) for i in range(5)])
        # "after lost observations" masks:
        self.T_i = np.sum(self.included[3], 1).reshape(
            (self.N, 1, 1))  # number of observations for each i
        # ensures minimum of 1 observation in order to avoid division error. If there are no observations, averages will be zero in any case
        self.T_i = self.T_i+(self.T_i <= 0)
        self.N_t = np.sum(self.included[3], 0).reshape(
            (1, self.max_T, 1))  # number of observations for each t
        # ensures minimum of 1 observation in order to avoid division error. If there are no observations, averages will be zero in any case
        self.N_t = self.N_t+(self.N_t <= 0)
        self.group_var_wght = 1-1/np.maximum(self.T_i-1, 1)

    def final_defs(self):
        self.W_a = self.W*self.a[3]
        self.tot_lost_obs = self.lost_obs*self.N
        self.NT = np.sum(self.included[3])
        self.NT_before_loss = self.NT+self.tot_lost_obs
        if not hasattr(self, 'n_dates'):
            self.number_of_RE_coef = 0
            self.number_of_RE_coef_in_variance = 0
            self.options.fixed_random_group_eff.set(0)
            self.options.fixed_random_time_eff.set(0)
            self.options.fixed_random_variance_eff.set(0)
        else:
            self.number_of_RE_coef = self.N * \
                (self.options.fixed_random_group_eff.value > 0) + \
                self.n_dates*(self.options.fixed_random_time_eff.value > 0)
            self.number_of_RE_coef_in_variance = (self.N*(self.options.fixed_random_group_eff.value > 0)
                                                  + self.n_dates*(self.options.fixed_random_time_eff.value > 0))*(self.options.fixed_random_variance_eff.value > 0)
        self.args = arguments.arguments(self)
        self.df = self.NT-self.args.n_args-self.number_of_RE_coef - \
            self.number_of_RE_coef_in_variance
        self.set_instrumentals()
        self.tobit()

    def lag_variables(self):
        T = self.max_T
        d = self.pqdkm[2]
        self.I = np.diag(np.ones(T))
        # differencing operator:
        if d == 0:
            return
        L0 = np.diag(np.ones(T-1), -1)
        Ld = (self.I-L0)
        Ld_inv = np.tril(np.ones((T, T)))
        for i in range(1, d):
            Ld = fu.dot(self.I-L0, Ld)
            Ld_inv = np.cumsum(Ld_inv, 0)
        self.Ld_inv = Ld_inv
        # multiplication:
        self.X = self.lag_variable(self.X, Ld, d, True)
        self.Y = self.lag_variable(self.Y, Ld, d, False)
        if not self.Z is None:
            self.Z = self.lag_variable(self.Z, Ld, d, True)

    def lag_variable(self, X, Ld, d, recreate_intercept):
        X_out = fu.dot(Ld, X)*self.a[3]
        if self.input.has_intercept and recreate_intercept:
            X_out[:, :, 0] = 1
        X_out[:, :d] = 0
        return X_out

    def params_ok(self, args):
        a = self.q_sel, self.p_sel, self.M_sel, self.K_sel
        for i in a:
            if len(i) > 0:
                if np.any(np.abs(args[i]) > 0.999):
                    return False
        return True

    def set_instrumentals(self):
        if self.input.Z.shape[1] == 1:
            self.XIV = self.X
        else:
            self.XIV = self.X = self.Z
        return
        ll = logl.LL(self.args.args_init, self)
        ll.standardize(self)
        Z_st, Z_st_long = ll.standardize_variable(panel, self.Z)
        ZZ = fu.dot(Z_st, Z_st)
        ZZInv = np.linalg.inv(ZZ)
        ZX = fu.dot(Z_st, ll.X_st)
        ZZInv_ZX = fu.dot(ZZInv, ZX)
        # using non-normalized first, since XIV should be unnormalized.
        self.XIV = fu.dot(self.Z, ZZInv_ZX)

    def subtract_means(self, X, Y, Z):
        subtract = self.options.subtract_means.value
        if not subtract:
            return X, Y, Z
        X = X-np.mean(X, 0)
        if self.input.has_intercept:
            X[:, 0] = 1
        Y = Y-np.mean(Y, 0)
        if self.input.Z.shape[1] == 1:
            return X, Y, Z
        Z = Z-np.mean(Z, 0)
        Z[:, 0] = 1

    def is_single(self):
        IDs, t = self.input.IDs_num, self.input.timevar_num
        try:
            if (np.all(IDs.iloc[0] == IDs) or np.all(t.iloc[0] == t)):
                return True
        except:
            return True
        return False

    def arrayize(self):
        """Splits X and Y into an arry of equally sized matrixes rows equal to the largest for each IDs"""

        X, Y, W, IDs, timevar, Z = [to_numpy(i) for i in
                                    (self.input.X, self.input.Y, self.input.W, self.input.IDs_num, self.input.timevar_num, self.input.Z)]
        X, Y, Z = self.subtract_means(X, Y, Z)
        NT, k = X.shape
        self.total_obs = NT
        if self.is_single():
            if not np.all(timevar[:, 0] == np.sort(timevar[:, 0])):  # remove in production
                raise RuntimeError("The time variable is not sorted!?!")
            self.X = np.array(X.reshape((1, NT, k)))
            self.Y = np.array(Y.reshape((1, NT, 1)))
            NT, kW = W.shape
            self.W = np.array(W.reshape((1, NT, kW)))
            NT, kZ = Z.shape
            self.Z = None
            if Z.shape[1] > 1:  # instrumental variables used
                self.Z = np.array(Z.reshape((1, NT, kZ)))
            self.time_map = None
            self.map = np.arange(NT).reshape(1, NT)
            self.N = 1
            self.max_T = NT
            self.T_arr = np.array([[NT]])
            self.date_counter = np.arange(self.max_T).reshape((self.max_T, 1))
            included = np.array([(self.date_counter >= self.lost_obs) *
                                (self.date_counter < self.T_arr[i]) for i in range(self.N)])
            self.date_count_mtrx = None
            self.date_count = None
            self.idincl = np.array([True])
            if not np.all(timevar[:, 0] == np.sort(timevar[:, 0])):  # remove in production
                raise RuntimeError(
                    "The arrayize procedure has unsorted the time variable!?!")
        else:
            sel, ix = np.unique(IDs, return_index=True)
            N = len(sel)
            sel = (IDs.T == sel.reshape((N, 1)))
            T = np.sum(sel, 1)
            self.max_T = np.max(T)
            self.idincl = T > self.lost_obs+self.options.min_group_df.value
            self.X = arrayize(X, N, self.max_T, T, self.idincl, sel)
            self.Y = arrayize(Y, N, self.max_T, T, self.idincl, sel)
            self.W = arrayize(W, N, self.max_T, T, self.idincl, sel)
            self.Z = arrayize(Z, N, self.max_T, T, self.idincl, sel)
            self.N = np.sum(self.idincl)
            varmap = np.arange(NT).reshape(NT, 1)
            self.map = arrayize(varmap, N, self.max_T, T, self.idincl,
                                sel, dtype=int).reshape((self.N, self.max_T))
            self.T_arr = T[self.idincl].reshape((self.N, 1))
            self.date_counter = np.arange(self.max_T).reshape((self.max_T, 1))
            included = np.array([(self.date_counter >= self.lost_obs) *
                                (self.date_counter < self.T_arr[i]) for i in range(self.N)])
            if len(included) < 5:
                raise RuntimeError(
                    f"{len(included)} valid observations, cannot perform panel analysis. Try without panel (don't specify ID and time)")
            self.get_time_map(timevar, self.N, T, self.idincl, sel, included)

            if np.sum(self.idincl) < len(self.idincl):
                idname = self.input.IDs.columns[0]
                idremoved = np.array(self.input.IDs)[
                    ix, 0][self.idincl == False]
                s = formatarray(idremoved, 90, ', ')
                print(
                    f"The following {idname}s were removed because of insufficient observations:\n %s" % (s))
            # remove in production. Checking sorting:
            tvar = arrayize(timevar, N, self.max_T, T, self.idincl, sel)
            a = [tvar[i][0][0] for i in self.date_map]
            if not np.all(a == np.sort(a)):
                raise RuntimeError(
                    "It seems the data is not properly sorted on time")

        zeros = np.zeros((self.N, self.max_T, 1))
        ones = np.ones((self.N, self.max_T, 1))
        self.included = [None, None]
        self.zeros = [None, None]
        self.ones = [None, None]
        self.included.extend(
            [included.reshape(list(included.shape)[:-1]+[1]*i) for i in range(5)])
        self.zeros.extend(
            [zeros.reshape(list(zeros.shape)[:-1]+[1]*i) for i in range(5)])
        self.ones.extend([ones.reshape(list(ones.shape)[:-1]+[1]*i)
                         for i in range(5)])

    def tobit(self):
        """Sets the tobit threshold"""
        tobit_limits = self.options.tobit_limits.value
        if tobit_limits is None:
            return
        if len(tobit_limits) != 2:
            print(
                "Warning: The tobit_limits argument must have lenght 2, and be on the form [floor, ceiling]. None is used to indicate no active limit")
        if (not (tobit_limits[0] is None)) and (not (tobit_limits[1] is None)):
            if tobit_limits[0] > tobit_limits[1]:
                raise RuntimeError(
                    "floor>ceiling. The tobit_limits argument must have lenght 2, and be on the form [floor, ceiling]. None is used to indicate no active limit")
        g = [1, -1]
        self.tobit_I = [None, None]
        self.tobit_active = [False, False]  # lower/upper threshold
        desc = ['tobit_low', 'tobit_high']
        for i in [0, 1]:
            self.tobit_active[i] = not (tobit_limits[i] is None)
            if self.tobit_active[i]:
                if np.sum((g[i]*self.Y < g[i]*tobit_limits[i])*self.a[3]):
                    print(
                        "Warning: there are observations of self.Y outside the non-censored interval. These will be ignored.")
                I = (g[i]*self.Y <= g[i]*tobit_limits[i])*self.a[3]
                self.Y[I] = tobit_limits[i]
                if np.var(self.Y) == 0:
                    raise RuntimeError(
                        "Your tobit limits are too restrictive. All observationss would have been cencored. Cannot run regression with these limits.")
                self.tobit_I[i] = I
                # avoiding singularity #Not sure why this is here, shuld be deleted?
                if np.sum(I) > 0 and np.sum(I) < self.NT and False:
                    self.X = np.concatenate((self.X, I), 2)
                    self.input.X_names.append(desc[i])

    def get_time_map(self, timevar, N, T_count, idincl, sel, incl):
        if timevar is None:
            return None
        N, T, k = incl.shape
        unq, ix = np.unique(timevar, return_inverse=True)
        t = arrayize(np.array(ix).reshape((len(timevar), 1)),
                     N, self.max_T, T_count, idincl, sel, int)  # maps N,T -> unique date
        grp_cnt = incl*np.arange(N).reshape(N, 1, 1)
        t_cnt = incl*np.arange(T).reshape(1, T, 1)
        incl = incl[:, :, 0]
        t = np.concatenate((t[incl],  grp_cnt[incl], t_cnt[incl]), 1)
        a = np.argsort(t[:, 0])
        t = t[a]  # three columns: unique date index, group number, day sequence

        tid = t[:, 0]  # unique date
        t_map = [[] for i in range(np.max(tid)+1)]  # all possible unique dates
        for i in range(len(tid)):
            t_map[tid[i]].append(t[i, 1:])  # appends group and day sequence
        t_map_tuple = []
        tcnt = []
        self.date_count_mtrx = np.zeros((N, T, 1))
        for i in range(len(t_map)):
            # group and day sequence for unique date i
            a = np.array(t_map[i]).T
            if len(a):
                # group and day sequence reference tuple
                m = (tuple(a[0]), tuple(a[1]))
                n_t = len(a[0])  # number of groups at this unique date
                # group and day reference tuple for the data matrix, for each unique date
                t_map_tuple.append(m)
                tcnt.append(n_t)  # count of groups at each date
                # timeseries matrix of the group count
                self.date_count_mtrx[m] = n_t

        # A full random effects calculation is infeasible because of complexity and computing costs.
        # A quazi random effects weighting is used. It  is more conservative than the full
        # RE weight theta=1-sd_pooled/(sd_pooled+sd_within/T)**0.5
        # If the weights are too generous, the RE adjustment may add in stead of reducing noise.
        n = len(tcnt)
        self.n_dates = n
        self.date_count = np.array(tcnt).reshape(n, 1, 1)
        self.date_map = t_map_tuple

    def get_time_map2(self, timevar, N, T_count, idincl, sel):
        if timevar is None:
            return None
        unq, ix = np.unique(timevar, return_inverse=True)
        n_dates = len(unq)
        t = arrayize(np.array(ix).reshape((n_dates, 1)), N, self.max_T,
                     T_count, idincl, sel, int)  # maps N,T -> unique date
        N, T, k = t.shape

        t_map = [[] for i in range(n_dates)]  # all possible unique dates
        for i in range(len(tid)):
            t_map[tid[i]].append(t[i, 1:])  # appends group and day sequence
        t_map_tuple = []
        tcnt = []
        self.date_count_mtrx = np.zeros((N, T, 1))
        for i in range(len(t_map)):
            # group and day sequence for unique date i
            a = np.array(t_map[i]).T
            if len(a):
                # group and day sequence reference tuple
                m = (tuple(a[0]), tuple(a[1]))
                n_t = len(a[0])  # number of groups at this unique date
                # group and day sequence reference tuple, for each unique date
                t_map_tuple.append(m)
                tcnt.append(n_t)  # count of groups at each date
                # timeseries matrix of the group count
                self.date_count_mtrx[m] = n_t

        # A full random effects calculation is infeasible because of complexity and computing costs.
        # A quazi random effects weighting is used. It  is more conservative than the full
        # RE weight theta=1-sd_pooled/(sd_pooled+sd_within/T)**0.5
        # If the weights are too generous, the RE adjustment may add in stead of reducing noise.
        n = len(tcnt)
        self.n_dates = n
        self.date_count = np.array(tcnt).reshape(n, 1, 1)
        self.date_map = t_map_tuple

    def define_h_func(self):
        h_def = """
def h(e,z):
	e2			=	e**2+1e-5
	h_val		=	np.log(e2)	
	h_e_val		=	2*e/e2
	h_2e_val	=	2/e2-4*e**2/e2**2

	return h_val,h_e_val,h_2e_val,None,None,None
		"""
        h_definition = self.options.h_function.value
        if h_definition is None:
            h_definition = h_def
        d = dict()
        try:
            exec(h_definition, globals(), d)
            ret = d['h'](1, 1)
            if len(ret) != 6:
                raise RuntimeError("""Your custom h-function must return exactly six arguments
				(h, dh/dx and ddh/dxdx for both e and z. the z return values can be set to None)""")
            self.h_def = h_definition
        except Exception as e:
            print(
                'Something is wrong with your custom function, default is used:' + str(e))
            exec(h_def, globals(), d)
            self.h_def = h_def

        self.z_active = True
        for i in ret[3:]:
            self.z_active = self.z_active and not (i is None)

    def mean(self, X, axis=None):
        dims = list(X.shape)
        dims[2:] = [1]*(len(dims)-2)
        # X=X*self.included.reshape(dims)
        if axis is None:
            return np.sum(X)/self.NT
        if axis == 1:
            dims.pop(1)
            return np.sum(X, 1)/self.T_i.reshape(dims)
        if axis == 0:
            dims.pop(0)
            return np.sum(X, 0)/self.N_t.reshape(dims)
        if axis == (0, 1):
            return np.sum(np.sum(X, 0), 0)/self.NT

    def var(self, X, axis=None, k=1, mean=None, included=None):
        dims = list(X.shape)
        dims_m = np.array(X.shape)
        dims[2:] = [1]*(len(dims)-2)
        if included is None:
            a = self.included[len(dims)]
        else:
            a = included
        if mean is None:
            m = self.mean(X*a, axis)
        else:
            m = mean

        if axis == None:
            Xm = (X-m)*a
            return np.sum(Xm**2)/(self.NT-k)

        if axis == 1:
            dims_m[1] = 1
            m = m.reshape(dims_m)
            Xm = (X-m)*a
            dims.pop(1)
            return np.sum((Xm)**2, 1)/np.maximum(self.T_i-k, 1).reshape(dims)
        if axis == 0:
            dims_m[0] = 1
            m = m.reshape(dims_m)
            Xm = (X-m)*a
            dims.pop(0)
            return np.sum((Xm)**2, 0)/np.maximum(self.N_t-k, 1).reshape(dims)
        if axis == (0, 1):
            dims_m[0:2] = 1
            m = m.reshape(dims_m)
            Xm = (X-m)*a
            return np.sum((Xm)**2, axis)/(self.NT-k)


def arrayize(X, N, max_T, T, idincl, sel, dtype=None):
    if X is None:
        return None
    NT, k = X.shape
    if dtype is None:
        Xarr = np.zeros((N, max_T, k))
    else:
        Xarr = np.zeros((N, max_T, k), dtype=dtype)
    T_used = []
    k = 0
    for i in range(len(sel)):
        if idincl[i]:
            Xarr[k, :T[i]] = X[sel[i]]
            k += 1
    Xarr = Xarr[:k]
    return Xarr


def to_numpy(x):
    x = np.array(x)
    if len(x.shape) == 2:
        return x
    return x.reshape((len(x), 1))


def formatarray(array, linelen, sep):
    s = sep.join([str(i) for i in array])
    s = '\n'.join(s[n:n + linelen] for n in range(0, len(s), linelen))
    return s
