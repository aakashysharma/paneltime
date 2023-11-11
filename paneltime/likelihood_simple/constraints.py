#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from ..output import stat_functions as stat
import os
path = os.path.dirname(__file__)


class Constraint:
    def __init__(self, index, assco, cause, value, interval, names, category):
        self.name = names[index]
        self.intervalbound = None
        self.max = None
        self.min = None
        self.value = None
        self.value_str = None
        if interval is None:
            self.value = value
            self.value_str = str(round(self.value, 8))
        else:
            if interval[0] > interval[1]:
                raise RuntimeError('Lower constraint cannot exceed upper')
            self.min = interval[0]
            self.max = interval[1]
            self.cause = 'user/general constraint'
        self.assco_ix = assco
        if assco is None:
            self.assco_name = None
        else:
            self.assco_name = names[assco]
        self.cause = cause
        self.category = category


class Constraints(dict):

    """Stores the constraints of the LL maximization"""

    def __init__(self, panel, args):
        dict.__init__(self)
        self.categories = {}
        self.fixed = {}
        self.intervals = {}
        self.associates = {}
        self.collinears = {}
        self.weak_mc_dict = {}
        self.args = args
        self.args[0]
        self.panel_args = panel.args
        self.CI = None
        self.pqdkm = panel.pqdkm
        self.m_zero = panel.m_zero
        self.ARMA_constraint = panel.options.ARMA_constraint.value
        self.H_correl_problem = False
        self.mc_problems = []

    def add(self, name, assco, cause, interval=None, replace=True, value=None, args=None):
        # (self,index,assco,cause,interval=None,replace=True,value=None)
        name, index = self.panel_args.get_name_ix(name)
        name_assco, assco = self.panel_args.get_name_ix(assco, True)
        for i in index:
            self.add_item(i, assco, cause, interval, replace, value, args)

    def add_item(self, index, assco, cause, interval, replace, value, args):
        """Adds a constraint. 'index' is the position
        for which the constraints shall apply.  \n\n

        Equality constraints are chosen by specifying 'minimum_or_value' \n\n
        Inequality constraints are chosen specifiying 'maximum' and 'minimum'\n\n
        'replace' determines whether an existing constraint shall be replaced or not 
        (only one equality and inequality allowed per position)"""

        if args is None:
            args = self.panel_args

        if not replace:
            if index in self:
                return False
        interval, value = test_interval(interval, value)
        if interval is None:  # this is a fixed constraint
            if len(self.fixed) == len(args.caption_v)-1:  # can't lock all variables
                return False
            if value is None:
                value = self.args[index]
            if index in self.intervals:
                c = self[index]
                if not (c.min < value < c.max):
                    return False
                else:
                    self.intervals.pop(index)
        elif index in self.fixed:  # this is an interval constraint, no longer a fixed constraint
            self.fixed.pop(index)

        eq, category, j = self.panel_args.positions_map[index]
        if not category in self.categories:
            self.categories[category] = [index]
        elif not index in self.categories[category]:
            self.categories[category].append(index)

        c = Constraint(index, assco, cause, value,
                       interval, args.caption_v, category)
        self[index] = c
        if value is None:
            self.intervals[index] = c
        else:
            self.fixed[index] = c
        if not assco is None:
            if not assco in self.associates:
                self.associates[assco] = [index]
            elif not index in self.associates[assco]:
                self.associates[assco].append(index)
        if cause == 'collinear':
            self.collinears[index] = assco
        return True

    def set_fixed(self, x):
        """Sets all elements of x that has fixed constraints to the constraint value"""
        for i in self.fixed:
            x[i] = self.fixed[i].value

    def within(self, x, fix=False):
        """Checks if x is within interval constraints. If fix=True, then elements of
        x outside constraints are set to the nearest constraint. if fix=False, the function 
        returns False if x is within constraints and True otherwise"""
        for i in self.intervals:
            c = self.intervals[i]
            if (c.min <= x[i] <= c.max):
                c.intervalbound = None
            else:
                if fix:
                    x[i] = max((min((x[i], c.max)), c.min))
                    c.intervalbound = str(round(x[i], 8))
                else:
                    return False
        return True

    def add_static_constraints(self, panel, its, ll=None, minvarhits=[]):
        pargs = self.panel_args
        p, q, d, k, m = self.pqdkm

        if its < -4:
            c = 0.5
        else:
            c = self.ARMA_constraint

        general_constraints = [
            ('rho', -c, c), ('lambda', -c, c), ('gamma', -c, c), ('psi', -c, c)]
        self.add_custom_constraints(panel, general_constraints, ll)
        self.add_custom_constraints(panel, pargs.user_constraints, ll)
        if len(minvarhits):
            self.add('gamma', None, 'Variance under threshold for an observation')
            self.add('psi', None, 'Variance under threshold for an observation')
            if not ll is None:
                for i in range(1, len(ll.args.args_d['omega'])):
                    self.add(
                        f'omega{i}', None, 'Variance under threshold for an observation')

        c = [(0, 0, 0, 0),
             (0, 0, 0, 0),
             (1, 0, 0, 0),
             (0, 1, 0, 0),
             (0, 0, 1, 0),
             (0, 0, 0, 1),
             (0, 0, 1, 1),
             (1, 1, 0, 0),
             (1, 1, 1, 1),

             ]
        if its < len(c):
            constr = self.get_init_constr(*c[its])
            self.add_custom_constraints(panel, constr, ll)
        a = 0

    def get_init_constr(self, p0, q0, k0, m0):
        p, q, d, k, m = self.pqdkm
        constr_list = ([(f'rho{i}', None) for i in range(p0, p)] +
                       [(f'lambda{i}', None) for i in range(q0, q)] +
                       [(f'gamma{i}', None) for i in range(k0, k)] +
                       [(f'psi{i}', None) for i in range(m0, m)])
        return constr_list

    def add_dynamic_constraints(self, panel, H, ll, args=None):
        if not args is None:
            self.args = args
            self.args[0]
        k, k = H.shape
        self.weak_mc_dict = dict()
        include = np.array(k*[True])
        include[list(self.fixed)] = False
        self.mc_problems = []  # list of [index,associate,condition index]
        self.CI = constraint_multicoll(k, panel, include, self.mc_problems, H)
        self.add_mc_constraint(panel)

    def add_custom_constraints(self, panel, constraints, ll, replace=True, cause='user constraint', clear=False, args=None):
        """Adds custom range constraints\n\n
            constraints shall be on the format [(name, minimum, maximum), ...]
            or
            a dictionary of the same form of an argument dictionary
            (a dictionary of dictonaries on the form dict[category][name]) 
            where items are lists of [minimum,maximum] or the fixing value

            name is taken from self.panel_args.caption_v"""

        if clear:
            self.clear(cause)
        if type(constraints) == list or type(constraints) == tuple:
            for c in constraints:
                self.add_custom_constraints_list(c, replace, cause, args, ll)
        elif type(constraints) == dict:
            self.add_custom_constraints_dict(
                panel, constraints, replace, cause, args)
        else:
            raise TypeError("The constraints must be a list, tuple or dict.")

    def add_custom_constraints_list(self, constraint, replace, cause, args, ll):
        """Adds a custom range constraint\n\n
            constraint shall be on the format (name, minimum, maximum)"""
        if type(constraint) == str or isinstance(constraint, int):
            name, value = constraint, None
            self.add(name, None, cause, replace=replace,
                     value=value, args=args)
            return
        elif len(constraint) == 2:
            name, minimum, maximum = list(constraint)+[None]
        elif len(constraint) == 3:
            name, minimum, maximum = constraint
        else:
            raise TypeError(
                "A constraint needs to be a string, integer or iterable with lenght no more than tree.")

        name, ix = self.panel_args.get_name_ix(name)
        m = [minimum, maximum]
        for i in range(2):
            if type(m[i]) == str:
                try:
                    m[i] = eval(m[i], globals(), ll.__dict__)
                except:
                    print(f"Custom constraint {name} ({m[i]}) failed")
                    return
        [minimum, maximum] = m
        self.add(name, None, cause, [minimum, maximum], replace, args=args)

    def add_custom_constraints_dict(self, panel, constraints, replace, cause, args):
        """Adds a custom range constraint\n\n
           If list, constraint shall be on the format (minimum, maximum)"""
        for grp in constraints:
            for i in range(len(panel.args.caption_d[grp])):
                name = panel.args.caption_d[grp][i]
                c = constraints[grp]
                try:
                    iter(c)
                    self.add(name, None, cause, c, replace, args=args)
                except TypeError as e:
                    self.add(name, None, cause, [c, None], replace, args=args)

    def add_mc_constraint(self, panel):
        """Adds constraints for severe MC problems"""

        if len(self.mc_problems) == 0:
            return

        no_check = get_no_check(panel)
        a = [i[0] for i in self.mc_problems]
        if no_check in a:
            mc = self.mc_problems[a.index(no_check)]
            mc[0], mc[1] = mc[1], mc[0]
        mc_limit = panel.options.multicoll_threshold_report.value
        coll_max_limit = panel.options.multicoll_threshold_max.value
        for index, assc, cond_index in self.mc_problems:
            cond = not ((index in self.associates) or (
                index in self.collinears)) and (not index == no_check)
            # same condition, but possible have a laxer condition for weak_mc_dict.
            # contains also collinear variables that are only slightly collinear, which shall be restricted when calcuating CV-matrix:
            if cond and cond_index > mc_limit:
                self.weak_mc_dict[index] = [assc, cond_index]
            if cond and cond_index > coll_max_limit:  # adding restrictions:
                self.add(assc, index, 'collinear')
                return

    def print_constraints(self, kind=None):
        for desc, obj in [('All', self),
                          ('Fixed', self.fixed),
                          ('Intervals', self.intervals)]:
            print(f"{desc} constraints:")
            for i in obj:
                c = obj[i]
                try:
                    print(
                        f"constraint: {i}, associate:{c.assco_ix}, max:{c.max}, min:{c.min}, value:{c.value}")
                except:
                    print(
                        f"constraint: {i}, associate:{c.assco_ix}, max:{None}, min:{None}, value:{c.value}")


def test_interval(interval, value):
    if not interval is None:
        if np.any([i is None for i in interval]):
            if interval[0] is None:
                value = interval[1]
            else:
                value = interval[0]
            interval = None
    return interval, value


def normalize(H, include):
    C = -H[include][:, include]
    d = np.maximum(np.diag(C).reshape((len(C), 1)), 1e-30)**0.5
    C = C/(d*d.T)
    includemap = np.arange(len(include))[include]
    return C, includemap


def decomposition(H, include=None):
    if include is None:
        include = [True]*len(H)
    C, includemap = normalize(H, include)
    c_index, var_prop, d, p = stat.var_decomposition(XXNorm=C)
    c_index = c_index.flatten()
    return c_index, var_prop, includemap


def multicoll_problems(panel, H, include, mc_problems):
    c_index, var_prop, includemap = decomposition(H, include)
    MAX_VAR_PROP_FRACTION = 0.4
    if c_index is None:
        return False, False
    mc_list = []
    largest_ci = None
    limit = panel.options.multicoll_threshold_report.value
    for cix in range(1, len(c_index)):
        if (np.sum(var_prop[-cix] > MAX_VAR_PROP_FRACTION) > 1) and (c_index[-cix] > limit):
            if largest_ci is None:
                largest_ci = c_index[-cix]
            var_prop_ix = np.argsort(var_prop[-cix])[::-1]
            var_prop_val = var_prop[-cix][var_prop_ix]
            j = var_prop_ix[0]
            j = includemap[j]
            done = var_prop_check(
                var_prop_ix, var_prop_val, includemap, j, mc_problems, c_index[-cix], mc_list)
            if done:
                break
    return c_index[-1], mc_list


def var_prop_check(var_prop_ix, var_prop_val, includemap, assc, mc_problems, cond_index, mc_list):
    i = 1
    if var_prop_val[i] < 0.5:
        return True
    index = var_prop_ix[i]
    index = includemap[index]
    mc_problems.append([index, assc, cond_index])
    mc_list.append(index)
    return False


def get_no_check(panel):
    no_check = panel.options.do_not_constrain.value
    X_names = panel.input.X_names
    if not no_check is None:
        if no_check in X_names:
            return X_names.index(no_check)
        print("A variable was set for the 'Do not constraint' option (do_not_constrain), but it is not among the x-variables")


def constraint_multicoll(k, panel, include, mc_problems, H):
    CI_max = 0
    for i in range(k-1):
        CI, mc_list = multicoll_problems(panel, H, include, mc_problems)
        CI_max = max((CI_max, CI))
        if len(mc_list) == 0:
            break
        include[mc_list] = False
    return CI_max
