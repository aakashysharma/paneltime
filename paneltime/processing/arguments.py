#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This module contains the argument class for the panel object
from ..output import stat_functions as stat
from .. import system_settings
if system_settings.cython:
    from .. import likelihood_cython as logl
else:
    from .. import likelihood as logl

from .. import random_effects as re
from .. import functions as fu

import numpy as np


class arguments:
    """Sets initial arguments and stores static properties of the arguments"""

    def __init__(self, panel):
        p, q, d, k, m = panel.pqdkm
        self.initial_user_defined = False
        self.categories = ['beta', 'rho', 'lambda', 'gamma', 'psi', 'omega']
        if panel.options.include_initvar.value:
            self.categories += ['initvar']
        if panel.z_active:
            self.categories += ['z']
        self.mu_removed = True
        if not self.mu_removed:
            self.categories += ['mu']
        self.make_namevector(panel, p, q, k, m)
        initargs = self.initargs(p, d, q, m, k, panel)
        self.position_defs(initargs)
        self.set_init_args(panel, initargs)
        self.get_user_constraints(panel)

    def initvar_asignment(self, initargs, v, panel, rho, lmbda, beta, default):

        p, q, d, k, m = panel.pqdkm

        if panel.options.EGARCH.value == 0:
            initargs['omega'][0] = v
        else:
            initargs['omega'][0] = np.log(v)

        if panel.options.include_initvar.value:
            initargs['initvar'][0] = v
        if k > 0:
            initargs['omega'][0] = initargs['omega'][0]*0.05

        initargs['beta'] = beta

        if default:  # turns out calculating ARMA parameters serves no purpose, default is therefore allways true. These coefficients are easily found by linesearch for given regression coefficients.
            if q > 0:
                initargs['lambda'][0] = lmbda

            if p > 0:
                initargs['rho'][0] = rho

            if k > 0:
                initargs['gamma'][0] = 0.5

            if m > 0:
                initargs['psi'][0] = 0.5

    def get_user_constraints(self, panel):
        e = "User contraints must be a dict of dicts or a string evaluating to that, on the form of ll.args.dict_string. User constraints not applied"
        if type(panel.options.user_constraints.value) == dict:
            self.user_constraints = panel.options.user_constraints.value
        else:
            if panel.options.user_constraints.value is None or panel.options.user_constraints.value == '':
                self.user_constraints = {}
                return
            try:
                self.user_constraints = eval(
                    panel.options.user_constraints.value)
            except SyntaxError:
                print(f"Syntax error: {e}")
                self.user_constraints = {}
                return
            except Exception as e:
                print(e)
                self.user_constraints = {}
                return
        if not panel.z_active and 'z' in self.user_constraints:
            self.user_constraints.pop('z')
        if panel.options.include_initvar.value and 'initvar' in self.user_constraints:
            self.user_constraints.pop('initvar')
        for grp in list(self.user_constraints.keys()):
            try:
                self.user_constraints[grp] = np.array(
                    self.user_constraints[grp]).flatten()
                test = None
                if not self.user_constraints[grp][0] is None:
                    test = 1.0 * self.user_constraints[grp][0]
                if not self.user_constraints[grp][1] is None:
                    test = 1.0 * self.user_constraints[grp][1]
                if test is None:
                    print(f'Failed to apply user constraints for {grp}')
            except:
                print(f'Failed to apply user constraints for {grp}')
                self.user_constraints.pop(grp)

        for grp in self.user_constraints:
            if not grp in self.caption_d:
                print(
                    f"Constraint on {grp} not applied, {grp} not in arguments")

    def initargs(self, p, d, q, m, k, panel):

        args = dict()
        args['beta'] = np.zeros((panel.X.shape[2], 1))
        args['omega'] = np.zeros((panel.W.shape[2], 1))
        args['rho'] = np.zeros(p)
        args['lambda'] = np.zeros(q)
        args['psi'] = np.zeros(m)
        args['gamma'] = np.zeros(k)
        args['omega'][0][0] = 0
        args['mu'] = np.array([])
        if panel.options.include_initvar.value:
            args['initvar'] = np.zeros(1)
        args['z'] = np.array([])

        if m > 0 and panel.z_active:
            args['z'] = np.array([1e-09])

        if panel.N > 1 and not self.mu_removed:
            args['mu'] = np.array([0.0001])

        return args

    def set_init_args(self, panel, initargs=None, default=True):
        p, q, d, k, m = panel.pqdkm

        if initargs is None:
            initargs = self.initargs(p, d, q, m, k, panel)

        # de2=np.roll(e**2,1)-e**2
        # c=stat.correl(np.concatenate((np.roll(de2,1),de2),2),panel)[0,1]
        beta, omega = self.set_init_regression(initargs, panel, default)
        self.args_start = self.create_args(initargs, panel)
        self.args_init = self.create_args(initargs, panel)
        self.set_restricted_args(p, d, q, m, k, panel, omega, beta)
        self.n_args = len(self.args_init.args_v)

    def set_restricted_args(self, p, d, q, m, k, panel, omega, beta):
        args_restricted = self.initargs(p, d, q, m, k, panel)
        args_OLS = self.initargs(p, d, q, m, k, panel)

        args_restricted['beta'][0][0] = np.mean(panel.Y)
        args_OLS['beta'] = beta

        v = panel.var(panel.Y)
        if panel.options.EGARCH.value == 0:
            args_restricted['omega'][0][0] = v
            args_OLS['omega'][0][0] = omega
        else:
            args_restricted['omega'][0][0] = np.log(v)
            args_OLS['omega'][0][0] = np.log(omega)

        self.args_restricted = self.create_args(args_restricted, panel)
        self.args_OLS = self.create_args(args_OLS, panel)

    def create_null_ll(self, panel):
        if not hasattr(self, 'LL_OLS'):
            self.LL_OLS = logl.LL(self.args_OLS, panel).LL
            self.LL_null = logl.LL(self.args_restricted, panel).LL

    def position_defs(self, initargs):
        """Defines positions in vector argument"""

        self.positions = dict()
        # a dictionary of indicies containing the string name and sub-position of index within the category
        self.positions_map = dict()
        k = 0
        for i in self.categories:
            n = len(initargs[i])
            rng = range(k, k+n)
            self.positions[i] = rng
            for j in rng:
                # equation,category,relative position
                self.positions_map[j] = [0, i, j-k]
            k += n

    def conv_to_dict(self, args):
        """Converts a vector argument args to a dictionary argument. If args is a dict, it is returned unchanged"""
        if type(args) == dict:
            return args
        if type(args) == list:
            args = np.array(args)
        d = dict()
        k = 0
        for i in self.categories:
            n = len(self.positions[i])
            rng = range(k, k+n)
            d[i] = np.array(args[rng])
            if i == 'beta' or i == 'omega':
                d[i] = d[i].reshape((n, 1))
            k += n
        return d

    def conv_to_vector(self, args):
        """Converts a dict argument args to vector argument. if args is a vector, it is returned unchanged.\n
        If args=None, the vector of self.args_init is returned"""
        if type(args) == list or type(args) == np.ndarray:
            return np.array(args)
        v = np.array([])
        for i in self.categories:
            s = np.array(args[i])
            if len(s.shape) == 2:
                s = s.flatten()
            if len(s) > 0:
                v = np.concatenate((v, s))
        return v

    def make_namevector(self, panel, p, q, k, m):
        """Creates a vector of the names of all regression varaibles, 
        including variables, ARIMA and GARCH terms. This defines the positions
        of the variables througout the estimation."""
        d, names_d = {}, {}
        captions = list(panel.input.X.keys())  # copy variable names
        d['beta'] = list(captions)
        c = [list(captions)]
        names = panel.input.X_names
        # names = [f'x{i}' for i in range(panel.n_beta)]
        names_d['beta'] = list(names)
        add_names(p, 'rho%s    AR    p', 'rho', d, c, captions, names, names_d)
        add_names(q, 'lambda%s MA    q', 'lambda',
                  d, c, captions, names, names_d)
        add_names(k, 'gamma%s  GARCH k', 'gamma',
                  d, c, captions, names, names_d)
        add_names(m, 'psi%s    ARCH  m', 'psi', d, c, captions, names, names_d)

        d['omega'] = list(panel.input.W.keys())
        captions.extend(panel.input.W.keys())

        names_d['omega'] = [f'omega{i}' for i in range(panel.nW)]
        names.extend(names_d['omega'])

        c.append(d['omega'])
        if panel.options.include_initvar.value:
            d['initvar'] = ['Initial variance']
            captions.extend(d['initvar'])
            names_d['initvar'] = d['initvar']
            names.extend(names_d['initvar'])
            c.append(d['initvar'])

        if m > 0:
            if panel.N > 1 and not self.mu_removed:
                d['mu'] = ['mu (var.ID eff.)']
                captions.extend(d['mu'])
                names_d['mu'] = ['mu']
                names.extend(d['mu'])
                c.append(d['mu'])
            if panel.z_active:
                d['z'] = ['z in h(e,z)']
                captions.extend(d['z'])
                names_d['z'] = ['z']
                names.extend(d['z'])
                c.append(d['z'])

        self.caption_v = captions
        self.caption_d = d
        self.names_v = names
        self.names_d = names_d
        self.names_category_list = c

    def create_args(self, args, panel, constraints=None):
        if isinstance(args, arguments_set):
            self.test_consistency(args)
            return args
        args_v = self.conv_to_vector(args)
        if not constraints is None:
            constraints.within(args_v, True)
            constraints.set_fixed(args_v)
        args_d = self.conv_to_dict(args_v)
        dict_string = []
        for c in self.categories:
            s = []
            captions = self.caption_d[c]
            a = args_d[c].flatten()
            for i in range(len(captions)):
                s.append(f"'{captions[i]}':{a[i]}")
            dict_string.append(f"'{c}':\n"+"{"+",\n".join(s)+"}")
        dict_string = "{"+",\n".join(dict_string)+"}"
        return arguments_set(args_d, args_v, dict_string, self, panel)

    def test_consistency(self, args):
        # for debugging only
        m = self.positions_map
        for i in m:
            dict_arg = args.args_d[m[i][1]]
            if len(dict_arg.shape) == 2:
                dict_arg = dict_arg[m[i][2]]
            if dict_arg[0] != args.args_v[i]:
                raise RuntimeError("argument inconsistency")

    def get_name_ix(self, x, single_item=False):
        # returns name, list of indicies
        if x is None:
            return None, None
        if x in self.caption_v:
            if single_item:
                indicies = self.caption_v.index(x)
            else:
                indicies = [self.caption_v.index(x)]
            return x, indicies
        elif x in self.positions and not single_item:
            indicies = list(self.positions[x])
            return x, indicies
        elif x in self.names_v:
            if single_item:
                indicies = self.names_v.index(x)
            else:
                indicies = [self.names_v.index(x)]
            return x, indicies
        try:
            name = self.caption_v[x]
        except Exception as e:
            raise RuntimeError(
                f"{e}. The identifier of an argument must be an integer or a string macthing a name in 'self.caption_v' or a category in 'self.positions'")
        if single_item:
            return name, x
        else:
            return name, [x]

    def set_init_regression(self, initargs, panel, default):
        usrargs = panel.options.arguments.value
        beta, rho, lmbda, u = ARMA_regression(panel)
        self.init_var = panel.var(u)

        if not usrargs is None:
            if type(usrargs) == str:
                try:
                    usrargs = eval(usrargs.replace(
                        " array", " np.array").replace(', dtype=float64', ''))
                except NameError as e:
                    if str(e) == "name 'array' is not defined":
                        usrargs = eval(usrargs.replace("array", " np.array"))
            args = self.create_args(usrargs, panel)
            for c in args.args_d:
                initargs[c] = args.args_d[c]
            self.initial_user_defined = True

            return initargs['beta'], initargs['omega'][0, 0]

        if panel.options.fixed_random_variance_eff.value == 0:
            if self.init_var < 1e-20:
                print(
                    'Warning, your model may be over determined. Check that you do not have the dependent among the independents')

        self.initvar_asignment(initargs, self.init_var,
                               panel, rho, lmbda, beta, default)

        return beta, self.init_var


def set_GARCH(panel, initargs, u, m):
    matrices = logl.set_garch_arch(panel, initargs)
    if matrices is None:
        e = u
    else:
        AMA_1, AMA_1AR, GAR_1, GAR_1MA = matrices
        e = fu.dot(AMA_1AR, u)*panel.included[3]
    h = h_func(e, panel, initargs)
    if m > 0:
        initargs['gamma'][0] = 0
        initargs['psi'][0] = 0


def h_func(e, panel, initargs):
    z = None
    if len(initargs['z']) > 0:
        z = initargs['z'][0]
    h_val, h_e_val, h_2e_val, h_z, h_2z, h_e_z = logl.h(e, z, panel)
    return h_val*panel.included[3]


def ARMA_regression(panel):
    gfre = panel.options.fixed_random_group_eff.value
    tfre = panel.options.fixed_random_time_eff.value
    re_obj_i = re.re_obj(panel, True, panel.T_i, panel.T_i, gfre)
    re_obj_t = re.re_obj(
        panel, False, panel.date_count_mtrx, panel.date_count, tfre)
    X = (panel.X+re_obj_i.RE(panel.X, panel) +
         re_obj_t.RE(panel.X, panel))*panel.included[3]
    Y = (panel.Y+re_obj_i.RE(panel.Y, panel) +
         re_obj_t.RE(panel.Y, panel))*panel.included[3]
    beta, u = stat.OLS(panel, X, Y, return_e=True)
    rho, lmbda = ARMA_process_calc(u, panel)
    return beta, rho, lmbda, u


def ARMA_process_calc(e, panel):
    c = stat.correlogram(panel, e, 2, center=True)[1:]
    if abs(c[0]) < 0.1:
        return 0, 0

    rho = 0.5*(c[0] + c[1]/c[0])
    rho = max(min(rho, 0.99), -0.99)

    lmbda = 0
    den = 2*(c[0]-rho)
    rtexp = ((rho**2 - 1)*(rho**2 - 1 + 4*c[0]**2 - 4*c[0]*rho))
    if den != 0 and rtexp > 0:
        lmbda1 = (1 - 2*c[0]*rho + rho**2)/den
        lmbda2 = (rtexp**0.5) / den

        if abs(lmbda1+lmbda2) > abs(lmbda1-lmbda2):
            lmbda = max(min(lmbda1 - lmbda2, 0.99), -0.99)
        else:
            lmbda = max(min(lmbda1 + lmbda2, 0.99), -0.99)

    return rho, lmbda


def add_names(T, captionstr, category, d, c, captions, names, names_d):
    a = []
    n = []
    if ' ' in captionstr:
        namestr = captionstr.split(' ')[0]
    for i in range(T):
        a.append(captionstr % (i,))
        n.append(namestr % (i,))
    captions.extend(a)
    names.extend(n)
    d[category] = a
    names_d[category] = n
    c.append(a)


class arguments_set:
    """A class that contains the numeric arguments used in the maximization
    in all shapes and forms needed."""

    def __init__(self, args_d, args_v, dict_string, arguments, panel):
        self.args_d = args_d  # dictionary of arguments
        self.args_v = args_v  # vector of arguments
        # a string defining a dictionary of named arguments. For user input of initial arguments
        self.dict_string = dict_string
        self.caption_v = arguments.caption_v  # vector of captions
        self.caption_d = arguments.caption_d  # dict of captions
        self.names_v = arguments.names_v  # vector of names
        self.names_d = arguments.names_d  # dict of names
        self.n_args = len(self.args_v)
        self.pqdkm = panel.pqdkm
        self.positions = arguments.positions
        self.names_category_list = arguments.names_category_list
