#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This module calculates statistics and saves it to a file

from . import stat_functions as stat
from . import stat_dist
import textwrap

import numpy as np
import time
from datetime import datetime

STANDARD_LENGTH = 8


class Output:
    # This class handles auxilliary statistics (regression info, diagnostics, df accounting)
    def __init__(self, ll, panel, dx_norm):
        self.ll = ll
        self.panel = panel

        self.incr = 0
        self.dx_norm = dx_norm

        self.define_table_size()
        self.statistics_decimals = 3
        self.update(0, ll, self.incr, dx_norm)
        self.describe()

    def describe(self):
        s = f"{self.panel.input.Y_names[0]} = {' + '.join(self.panel.input.X_names)}"
        s = textwrap.TextWrapper(width=self.stat_totlen).fill(s) + '\n'
        if len(self.panel.input.Z_names[1:]):
            s += f"Instruments:\t{', '.join(self.panel.input.Z_names[1:])}\n"
        s = "Regression model:\n" + s
        self.model_desc = s

    def update(self, its, ll, incr, dx_norm):
        self.iterations = its
        self.ll = ll
        self.dx_norm = dx_norm
        self.incr = incr
        self.stats = Statistics(self.ll, self.panel)

    def get_model(self):
        p, q, d, k, m = self.panel.pqdkm
        s = ""
        if d > 0:
            s += f"ARIMA{(p, d, q)}"
        elif (p*q):
            s += f"ARMA{(p,q)}"
        elif (p == 0) & (q > 0):
            s += f"MA{q}"
        elif (p > 0) & (q == 0):
            s += f"AR{p}"

        if (k*m):
            if s != '':
                s += '-'
            s += f"GARCH{(k,m)}"
        t = self.panel.options.fixed_random_time_eff.value
        i = self.panel.options.fixed_random_group_eff.value
        if t+i:
            if s != '':
                s += ' '
            if (t == 2) & (i == 2):
                s += '2-way RE'
            elif (t == 1) & (i == 1):
                s += '2-way FE'
            elif (t == 2) | (i == 2):
                s += 'FE'
            elif (t == 1) | (i == 1):
                s += 'RE'

        if s == '':
            s = 'OLS'
            method = 'Least Squares'
        else:
            method = 'Maximum Likelihood'

        return s, method

    def statistics(self, start_time):
        s = self.stats
        panel = self.panel
        heading = 'Statistics:'
        model, method = self.get_model()
        n, T, k = self.panel.X.shape
        rob = panel.options.robustcov_lags_statistics.value[1] > 0
        run_time = np.round(time.time()-start_time)

        c0 = (
            ('Dep. Variable:', panel.input.Y_names[0]),
            ('Model:', model),
            ('Method:', method),
            ('Date:', datetime.now().strftime('%a, %d %b %Y')),
            ('Time:', datetime.now().strftime('%H:%M:%S')),
            ('Run time:', run_time),
            ('Observations count:', panel.NT),
            ('Df Residuals:', s.df),
            ('Df Model:', k-1),
            ('Covariance Type:', rob)
        )

        c1 = (
            ('R-squared:', s.Rsq),
            ('Adj R-squared:', s.Rsqadj),
            ('F-statistic:', s.F),
            ('Prob (F-statistic):', s.F_p),
            ('Log-Likelihood:', self.ll.LL),
            ('AIC:', 2*k - 2*self.ll.LL),
            ('BIC:', panel.NT*k - 2*self.ll.LL),
            ('Panel groups:', n),
            ('Panel dates:', T),
            ('', '')
        )

        tbl = [(c0[i], c1[i]) for i in range(len(c0))]
        return heading + '\n' + self.parse_tbl(tbl)

    def diagnostics(self, constr):

        s = self.stats
        heading = 'Diagnostics:'
        CI, n_CI = self.get_CI(constr)
        c0 = (
            ('Omnibus:', s.Omnibus_st),
            ('Prob(Omnibus):', s.Omnibus_st),
            ('Skew:', s.skewness_st),
            ('Kurtosis:', s.kurtosis),
            ('', ''),
            ('', ''),
            ('ADF statistic:', s.ADF_stat),
            ('', '')
        )

        c1 = (
            ('Durbin-Watson:', s.DW),
            ('Jarque-Bera (JB):', s.JB_st),
            ('Prob(JB):', s.JB_prob_st),
            ('Cond. No.:', CI),
            ('Cond. var count.:', n_CI),
            ('', ''),
            ('ADF crit.val 1%:', s.c1),
            ('ADF crit.val 5%:', s.c5),
        )

        tbl = [(c0[i], c1[i]) for i in range(len(c0))]
        return heading + '\n' + self.parse_tbl(tbl)

    def df_accounting(self, panel):

        N, T, k = self.panel.X.shape
        heading = 'Df accounting:'
        samsum = [
            ('SAMPLE SIZE SUMMARY:', ''),
            ('Original sample size:', panel.orig_size),
            ('Sample size after removals:', panel.NT_before_loss),
            ('Degrees of freedom:', panel.df),
            ('Number of variables:', k),
            ('Number of groups:', N),
            ('Number of dates (maximum):', T)
        ]

        grpremv = [
            ('GROUP REMOVAL:', ''),
            ('Lost observations:', ''),
            ('A) ARIMA/GARCH:', panel.lost_obs),
            ('B) min obs (user preferences):',
             panel.options.min_group_df.value),
            ('Required observations (A+B):', panel.lost_obs +
             panel.options.min_group_df.value),

            ('Groups removed:', ''),
            ('A) total # of groups:', len(panel.idincl)),
            ('B) # of groups removed:', sum(panel.idincl == False)),
            ('# of groups remaining (A-B):', sum(panel.idincl == True)),
            ('# of observations removed:',
             panel.input.X.shape[0]-panel.NT_before_loss)
        ]

        df = [('DEGREES OF FREEDOM:', ''),
              ('A) sample size:', panel.NT_before_loss),
              ('B) lost to GARCH/ARIMA:', panel.tot_lost_obs),
              ('C) lost to FE/RE:', panel.number_of_RE_coef),
              ('D) coefficients in regr:', panel.args.n_args),
              ('Degrees of freedom (A-B-C-D):', panel.df)
              ]

        tbl = [
            (samsum[0], None),
            (samsum[1], samsum[4]),
            (samsum[2], samsum[5]),
            (samsum[3], samsum[6]),
            (None, None),
            (grpremv[0], df[0]),
            (grpremv[1], df[1]),
            (grpremv[2], df[2]),
            (grpremv[3], df[3]),
            (grpremv[4], df[4]),
            (None, df[5]),
            (grpremv[5], None),
            (grpremv[6], None),
            (grpremv[7], None),
            (grpremv[8], None),
            (grpremv[9], None)

        ]

        return heading+'\n' + self.parse_tbl(tbl)

    def define_table_size(self):
        "Defines the columns and width of the statistics tables"
        self.stat_left_col = max((16 + len(self.panel.input.Y_names[0]), 27))
        self.stat_right_col = 40
        self.stat_totlen = self.stat_left_col + self.stat_right_col + 2

    def parse_tbl(self, tbl):
        c = [self.stat_left_col, self.stat_right_col]
        fmt = self.format_tbl_itm

        for i in range(len(tbl)):
            for j in [0, 1]:
                itm = tbl[i][j]
                if itm is None:
                    itm = ['', '']
                l = sum([len(str(itm[k])) for k in [0, 1]]) + 2
                if l > c[j]:
                    c[j] = l

        line = "="*(sum(c)+2) + '\n'
        s = line
        for i in range(len(tbl)):
            for j in [0, 1]:
                if tbl[i][j] is None:
                    s += fmt('', '', c[j])
                else:
                    s += fmt(tbl[i][j][0], tbl[i][j][1], c[j])
                if j == 0:
                    s += '  '
                else:
                    s += '\n'
        s += line

        return s

    def format_tbl_itm(self, description, value, length):
        try:
            value = str(np.round(value, self.statistics_decimals))
        except:
            value = str(value)
        return "{:<{}}{}".format(description, length - len(value), value)

    def get_CI(self, constr):
        CI = 'None'
        n_CI = 'None'
        if not constr is None:
            if not constr.CI is None:
                CI = np.round(constr.CI)
            n_CI = len(constr.mc_problems)
        return CI, n_CI


class RegTableObj(dict):
    # This class handles the regression table itself
    def __init__(self, panel, ll, g, H, G, constr, dx_norm, model_desc):
        dict.__init__(self)

        self.Y_names = panel.input.Y_names
        self.X_names = panel.input.X_names
        self.args = ll.args.dict_string
        self.n_variables = panel.args.n_args
        self.lags = panel.options.robustcov_lags_statistics.value[1]
        self.footer = f"\nSignificance codes: '=0.1, *=0.05, **=0.01, ***=0.001,    |=collinear\n\n{ll.err_msg}"
        self.dx_norm = dx_norm
        self.t_stats(panel, ll, H, G, g, constr)
        self.constraints_formatting(panel, constr)
        self.model_desc = model_desc

    def t_stats(self, panel, ll, H, G, g, constr):
        self.d = {'names': np.array(panel.args.caption_v),
                  'count': range(self.n_variables),
                  'args': ll.args.args_v}
        d = self.d

        T = len(d['names'])
        if H is None:
            return
        d['se_robust'], d['se_st'] = sandwich(
            H, G, g, constr, panel, self.lags)
        d['se_robust_oposite'], d['se_st_oposite'] = sandwich(
            H, G, g, constr, panel, self.lags, oposite=True)
        d['se_robust'][np.isnan(
            d['se_robust'])] = d['se_robust_oposite'][np.isnan(d['se_robust'])]
        d['se_st'][np.isnan(d['se_st'])
                   ] = d['se_st_oposite'][np.isnan(d['se_st'])]

        no_nan = np.isnan(d['se_robust']) == False
        valid = no_nan
        valid[no_nan] = (d['se_robust'][no_nan] > 0)
        d['tstat'] = np.array(T*[np.nan])
        d['tsign'] = np.array(T*[np.nan])
        d['tstat'][valid] = d['args'][valid]/d['se_robust'][valid]
        d['tsign'][valid] = (
            1-stat_dist.tcdf(np.abs(d['tstat'][valid]), panel.df))  # Two sided tests
        d['sign_codes'] = get_sign_codes(d['tsign'])
        z = stat_dist.tinv025(panel.df)
        d['conf_low'] = d['args'] - z*d['se_robust']
        d['conf_high'] = d['args'] + z*d['se_robust']

    def constraints_formatting(self, panel, constr):
        weak_mc_dict = {}
        if not constr is None:
            weak_mc_dict = constr.weak_mc_dict
        d = self.d
        if not self.dx_norm is None:
            d['dx_norm'] = self.dx_norm
        T = len(d['names'])
        d['set_to'], d['assco'], d['cause'], d['multicoll'] = [
            '']*T, ['']*T, ['']*T, ['']*T
        if constr is None:
            return
        c = constr.fixed
        for i in c:
            d['set_to'][i] = c[i].value_str
            d['assco'][i] = c[i].assco_name
            d['cause'][i] = c[i].cause

        c = constr.intervals
        for i in c:
            if not c[i].intervalbound is None:
                d['set_to'][i] = c[i].intervalbound
                d['assco'][i] = 'NA'
                d['cause'][i] = c[i].cause

        for i in weak_mc_dict:  # adding associates of non-severe multicollinearity
            d['multicoll'][i] = '|'
            d['assco'][i] = panel.args.caption_v[weak_mc_dict[i][0]]

    def table(self, n_digits=3, brackets='(', fmt='NORMAL', stacked=True, show_direction=False, show_constraints=True, show_confidence=False):
        include_cols, llength = self.get_cols(
            stacked, show_direction, show_constraints, show_confidence)
        if fmt == 'INTERNAL':
            self.X = None
            return str(self.args), None
        self.include_cols = include_cols
        self.n_cols = len(include_cols)
        for a, l, is_string, name, neg, just, sep, default_digits in pr:
            self[a] = column(self.d, a, l, is_string, name, neg,
                             just, sep, default_digits, self.n_variables)
        self.X = self.output_matrix(n_digits, brackets)
        s = format_table(self.X, include_cols, fmt,
                         "Paneltime GARCH-ARIMA panel regression",
                         self.footer, self.model_desc)
        return s, llength

    def output_matrix(self, digits, brackets):
        structured = False
        for i in range(self.n_cols):
            if type(self.include_cols[i]) == list:
                structured = True
                break
        if structured:
            return self.output_matrix_structured(digits, brackets)
        else:
            return self.output_matrix_flat(digits, brackets)

    def output_matrix_structured(self, digits, brackets):
        X = [['']*self.n_cols for i in range(3*(self.n_variables+1)-1)]
        for i in range(self.n_cols):
            a = self.include_cols[i]
            if type(a) == list:
                h = self[a[0]].name.replace(':', ' ')
                if brackets == '[':
                    X[0][i] = f"{h}[{self[a[1]].name}]:"
                elif brackets == '(':
                    X[0][i] = f"{h}({self[a[1]].name}):"
                else:
                    X[0][i] = f"{h}/{self[a[1]].name}:"
                v = [self[a[j]].values(digits) for j in range(3)]
                for j in range(self.n_variables):
                    X[(j+1)*3-1][i] = v[0][j]
                    if brackets == '[':
                        X[(j+1)*3][i] = f"[{v[1][j]}]{v[2][j]}"
                    elif brackets == '(':
                        X[(j+1)*3][i] = f"({v[1][j]}){v[2][j]}"
                    else:
                        X[(j+1)*3][i] = f"{v[1][j]}{v[2][j]}"
            else:
                X[0][i] = self[a].name
                v = self[a].values(digits)
                for j in range(self.n_variables):
                    X[(j+1)*3-1][i] = v[j]
        return X

    def output_matrix_flat(self, digits, brackets):
        X = [['']*self.n_cols for i in range(self.n_variables+1)]
        for i in range(self.n_cols):
            a = self.include_cols[i]
            X[0][i] = self[a].name
            v = self[a].values(digits)
            for j in range(self.n_variables):
                X[j+1][i] = v[j]
        return X

    def get_cols(self, stacked,
                 show_direction,
                 show_constraints,
                 show_confidence):
        "prints a single regression"
        dx_col = []
        llength = 9
        if show_direction:
            dx_col = ['dx_norm']
        else:
            llength -= 1
        mcoll_col = []
        if show_constraints:
            mcoll_col = ['multicoll', 'assco', 'set_to', 'cause']
        conf_coll = []
        if show_confidence:
            conf_coll = ['conf_low', 'conf_high']
        else:
            llength -= 2
        if stacked:
            cols = ['count', 'names', ['args', 'se_robust', 'sign_codes']
                    ] + conf_coll + dx_col + ['tstat', 'tsign'] + mcoll_col
        else:
            cols = ['count', 'names', 'args', 'se_robust', 'sign_codes'] + \
                conf_coll + dx_col + ['tstat', 'tsign'] + mcoll_col
        return cols, llength


class column:
    def __init__(self, d, a, l, is_string, name, neg, just, sep, default_digits, n_variables):
        self.length = l
        self.is_string = is_string
        self.name = name
        self.default_digits = default_digits
        self.neg_allowed = neg
        self.justification = just
        self.tab_sep = sep
        self.n_variables = n_variables
        if a in d:
            self.exists = True
            self.input = d[a]
        else:
            self.exists = False
            self.input = [' - ']*self.n_variables

    def values(self, digits):
        try:
            if self.length is None:
                if digits == 'SCI':
                    return self.input
                else:
                    return np.round(self.input, digits)
            return np.round(self.input, self.length)
        except:
            if self.length is None:
                return self.input
            else:
                return np.array([str(i).ljust(self.length)[:self.length] for i in self.input])


def sandwich(H, G, g, constr, panel, lags, oposite=False, resize=True):
    H, G, idx = reduce_size(H, G, g, constr, oposite, resize)
    lags = lags+panel.lost_obs
    onlynans = np.array(len(idx)*[np.nan])
    try:
        hessin = np.linalg.inv(-H)
    except np.linalg.LinAlgError as e:
        print(e)
        return np.array(onlynans), np.array(onlynans)
    se_robust, se, V = stat.robust_se(panel, lags, hessin, G)
    se_robust, se, V = expand_x(se_robust, idx), expand_x(
        se, idx), expand_x(V, idx, True)
    if se_robust is None:
        se_robust = np.array(onlynans)
    if se is None:
        se = np.array(onlynans)
    return se_robust, se


def reduce_size(H, G, g, constr, oposite, resize):
    # this looks unneccessary complicated
    if constr is None:
        return H, G, np.ones(len(g), dtype=bool)
    if (G is None) or (H is None):
        return
    m = len(H)
    if not resize:
        return H, G, np.ones(m, dtype=bool)
    weak_mc_dict = constr.weak_mc_dict.keys()
    c = list(constr.fixed.keys())
    if oposite:
        weak_mc_dict = [constr.weak_mc_dict[i][0] for i in constr.weak_mc_dict]
        c = []
        for i in constr.fixed:
            if not constr.fixed[i].assco_ix is None:
                c.append(constr.fixed[i].assco_ix)
    if False:  # not sure why this is here
        for i in weak_mc_dict:
            if not i in c:
                c.append(i)
    idx = np.ones(m, dtype=bool)
    if len(c) > 0:  # removing fixed constraints from the matrix
        idx[c] = False
        H = H[idx][:, idx]
        G = G[:, :, idx]
    return H, G, idx


def expand_x(x, idx, matrix=False):
    x = np.real(x)
    m = len(idx)
    if matrix:
        x_full = np.zeros((m, m))
        x_full[:] = np.nan
        ref = np.arange(m)[idx]
        for i in range(len(x)):
            try:
                x_full[ref[i], idx] = x[i]
                x_full[idx, ref[i]] = x[i]
            except:
                a = 0
    else:
        x_full = np.zeros(m)
        x_full[:] = np.nan
        x_full[idx] = x
    return x_full


def get_sign_codes(tsign):
    sc = []
    for i in tsign:
        if np.isnan(i):
            sc.append(i)
        elif i < 0.001:
            sc.append('***')
        elif i < 0.01:
            sc.append('** ')
        elif i < 0.05:
            sc.append('*  ')
        elif i < 0.1:
            sc.append("'  ")
        else:
            sc.append('')
    sc = np.array(sc, dtype='<U3')
    return sc


def remove_illegal_signs(name):
    illegals = ['#', 	'<', 	'$', 	'+',
                '%', 	'>', 	'!', 	'`',
                '&', 	'*', 	'‘', 	'|',
                '{', 	'?', 	'“', 	'=',
                '}', 	'/', 	':',
                '\\', 	'b']
    for i in illegals:
        if i in name:
            name = name.replace(i, '_')
    return name


class Statistics:
    def __init__(self, ll, panel):
        ll.standardize(panel)
        self.df = panel.df
        self.N, self.T, self.k = panel.X.shape
        self.Rsq_st, self.Rsqadj_st, self.LL_ratio, self.LL_ratio_OLS_st, self.F_st, self.F_p_st = stat.goodness_of_fit(
            ll, True, panel)
        self.Rsq, self.Rsqadj, self.LL_ratio, self.LL_ratio_OLS, self.F, self.F_p = stat.goodness_of_fit(
            ll, False, panel)
        self.no_ac_prob, self.rhos, self.RSqAC = stat.breusch_godfrey_test(
            panel, ll, 10)
        self.DW, self.DW_no_panel = stat.DurbinWatson(panel, ll)
        self.JB_prob, self.JB, self.skewness, self.kurtosis, self.Omnibus, self.Omnibus_pval = stat.JB_normality_test(
            ll.u_long, panel)
        self.JB_prob_st, self.JB_st, self.skewness_st, self.kurtosis_st, self.Omnibus_st,  self.Omnibus_pval_st = stat.JB_normality_test(
            ll.e_RE_norm_centered_long, panel)
        self.ADF_stat, self.c1, self.c5 = stat.adf_test(panel, ll, 10)
        self.instruments = panel.input.Z_names[1:]
        self.pqdkm = panel.pqdkm


def get_tab_stops(X, f):

    m_len = f.measure("m")
    counter = 2*m_len
    tabs = [f"{counter}", 'numeric']
    r, c = np.array(X).shape
    for i in range(c):
        t = 1
        num_max = 0
        for j in range(r):
            s = str(X[j][i])
            if '.' in s:
                a = s.split('.')
                num_max = max((len(a[0]), num_max))
            t = max((f.measure(X[j][i])+(num_max+2)*m_len, t))
        counter += t
        tabs.extend([f"{counter}", 'numeric'])
    return tabs


l = STANDARD_LENGTH
# python variable name,	length,		is string,  display name,		neg. values,	justification	next tab space		round digits (None=no rounding,-1=set by user)
pr = [
    ['count',		2,			False,		'',					False,			'right', 		2,					None],
    ['names',		None,		True,		'Variable names',
     False,			'right', 		2, 					None],
    ['args',		None,		False,		'coef',			True,			'right', 		2, 					-1],
    ['se_robust',	None,		False,		'std err',
     True,			'right', 		3, 					-1],
    ['sign_codes',	5,			True,		'',					False,			'left', 		2, 					-1],
    ['dx_norm',		None,		False,		'direction',
     True,			'right', 		2, 					None],
    ['tstat',		2,			False,		't',			True,			'right', 		2, 					2],
    ['tsign',		None,		False,		'P>|t|',
     False,			'right', 		2, 					3],
    ['conf_low',		None,		False,
     '[0.025',			False,			'right', 		2, 					3],
    ['conf_high',		None,		False,		'0.975]',
     False,			'right', 		2, 					3],
    ['multicoll',	1,			True,		'',					False,			'left', 		2, 					None],
    ['assco',		20,			True,		'collinear with',
     False,			'center', 		2, 					None],
    ['set_to',		6,			True,		'set to',
     False,			'center', 		2, 					None],
    ['cause',		50,			True,		'cause',			False,			'right', 		2, 					None]]


def format_table(X, cols, fmt, heading, tail, mod):
    if fmt == 'NORMAL':
        return format_normal(X, [1], cols)+tail
    if fmt == 'LATEX':
        return format_latex(X, cols, heading)+tail
    if fmt == 'HTML':
        return format_html(X, cols, heading, mod)+tail
    if fmt == 'CONSOLE':
        return format_console(X)+tail


def format_normal(X, add_rows=[], cols=[]):
    p = ''
    if 'multicoll' in cols:
        constr_pos = cols.index('multicoll')+1
        p = "\t"*constr_pos+"constraints:".center(38)
    p += "\n"
    for i in range(len(X)):
        p += '\n'*(i in add_rows)
        p += '\n'
        for j in range(len(X[0])):
            p += f'\t{X[i][j]}'

    return p


def format_console(X):
    p = ''
    X = np.array(X)
    dcol = 2
    n, k = X.shape
    colwith = [max(
        [len(s) for s in X[:, i]])
        for i in range(k)
    ]
    for i in range(n):
        p += '\n'
        for j in range(k):
            if j == 0:
                p += f'{X[i][j]}'.ljust(3)
            elif j == 1:
                p += f'{X[i][j]}'.ljust(colwith[j]+dcol)
            else:
                p += f'{X[i][j]}'.rjust(colwith[j]+dcol)
    p = p.split('\n')
    n = len(p[1])
    p[0] = '='*n
    p = p[:2] + ['-'*n] + p[2:] + ['='*n]
    p = '\n'.join(p)
    p = 'Regression results:\n' + p
    return p


def format_latex(X, cols, heading):
    X = np.array(X, dtype='U128')
    n, k = X.shape
    p = """
\\begin{table}[ht]
\\caption{%s} 
\\centering
\\begin{tabular}{""" % (heading,)
    p += ' c'*k+' }\n\\hline'
    p += '\t'+' &\t'.join(X[0])+'\\\\\n\\hline\\hline'
    for i in range(1, len(X)):
        p += '\t' + ' &\t'.join(X[i])+'\\\\\n'
    p += """
\\hline %inserts single line
\\end{tabular}
\\label{table:nonlin} % is used to refer this table in the text
\\end{table}"""
    return p


def format_html(X, cols, heading, head, mod):
    X = np.array(X, dtype='U128')
    n, k = X.shape
    if head[-1:] == '\n':
        head = head[:-1]
    head = head.replace('\n', "</td></tr>\n<td class='h'>")
    head = head.replace('\t', "</td><td class='h'>")
    mod = mod.replace('\n', '<br>')
    spaces = '&nbsp'*4
    p = (f"<br><br><h1>{heading}</h1><br><br>"
         f"<p>{mod}<br>"
         f"<table class='head'></tr><td class='h'>{head}</td></table><br><br>\n\n"
         f"<p><table>\t<tr><th>"
         )
    p += ('\t</th><th>'+spaces).join(X[0])+'</th></tr>\n'
    for i in range(1, len(X)):
        p += '\t<tr><td>'+'\t</td><td>'.join(X[i])+'</td></tr>\n'
    p += '</table></p>'
    return p


alphabet = 'abcdefghijklmnopqrstuvwxyz'


class join_table(dict):
    """Creates a  joint table of several regressions with columns of the join_table_column class.
    See join_table_column for data handling."""

    def __init__(self, args, panel, varnames=[]):
        dict.__init__(self)
        self.names_category_list = list(
            [list(i) for i in args.names_category_list])  # making a copy
        k = 0
        for i in varnames:
            if i in self.names_category_list[0]:
                k = self.names_category_list[0].index(i)+1
            else:
                self.names_category_list[0].insert(k, i)
                k += 1
        self.caption_v = [
            itm for s in self.names_category_list for itm in s]  # flattening
        self.footer = f"\nSignificance codes: '=0.1, *=0.05, **=0.01, ***=0.001,    |=collinear"

    def update(self, ll, stats, desc, panel):
        if not desc in self:
            for i in range(len(ll.args.names_category_list)):
                for j in ll.args.names_category_list[i]:
                    if not j in self.names_category_list[i]:
                        self.names_category_list[i].append(j)
            self.caption_v = [
                itm for s in self.names_category_list for itm in s]  # flattening
        self[desc] = join_table_column(stats, ll, panel)

    def make_table(self, stacked, brackets, digits, caption):
        keys = list(self.keys())
        k = len(keys)
        n = len(self.caption_v)
        if stacked:
            X = [['' for j in range(2+k)] for i in range(4+3*n)]
            for i in range(n):
                X[3*i+1][1] = self.caption_v[i]
                X[3*i+1][0] = i
            X[1+3*n][1] = 'Log likelihood'
            X[2+3*n][1] = 'Degrees of freedom'
            X[3+3*n][1] = 'Adjusted R-squared'
        else:
            X = [['' for j in range(2+2*k)] for i in range(4+n)]
            for i in range(n):
                X[i+1][1] = self.caption_v[i]
                X[i+1][0] = i
            X[1+n][1] = 'Log likelihood'
            X[2+n][1] = 'Degrees of freedom'
            X[3+n][1] = 'Adjusted R-squared'
        for i in range(k):
            self.make_column(i, keys[i], X, stacked, brackets, digits, caption)
        s = format_normal(X, [1, (1+stacked*2)*n+1, (1+stacked*2)*n+4])
        s += self.footer
        max_mod = 0
        models = []
        for i in range(len(keys)):
            key = self[keys[i]]
            p, q, d, k, m = key.pqdkm
            models.append(f"\n{alphabet[i]}: {keys[i]}")
            max_mod = max(len(models[i]), max_mod)
        for i in range(len(keys)):
            s += models[i].ljust(max_mod+2)
            if len(key.instruments):
                s += f"\tInstruments: {key.instruments}"
            s += f"\tARIMA({p},{d},{q})-GARCH({k},{m})"
        return s, X

    def make_column(self, col, key, X, stacked, brackets, digits, caption):
        if not 'se_robust' in self[key].stats:
            return

        if caption == 'JOINED LONG':
            X[0][(2-stacked)*col+2] += f"{self[key].Y_name} ({alphabet[col]})"
        else:
            X[0][(2-stacked)*col+2] = alphabet[col]
        n = len(self.caption_v)
        m = len(self[key].args.caption_v)
        ix = [self.caption_v.index(i) for i in self[key].args.caption_v]
        se = np.round(self[key].stats['se_robust'], digits)
        sgn = self[key].stats['sign_codes']
        args = np.round(self[key].args.args_v, digits)
        if brackets == '[':
            se_sgn = [f"[{se[i]}]{sgn[i]}" for i in range(m)]
        elif brackets == '(':
            se_sgn = [f"({se[i]}){sgn[i]}" for i in range(m)]
        else:
            se_sgn = [f"{se[i]}{sgn[i]}" for i in range(m)]
        if stacked:
            for i in range(m):
                X[3*ix[i]+1][col+2] = args[i]
                X[3*ix[i]+2][col+2] = se_sgn[i]
            X[1+3*n][col+2] = round_sign_digits(self[key].LL, 5, 1)
            X[2+3*n][col+2] = self[key].df
            X[3+3*n][col+2] = f"{round(self[key].Rsqadj*100,1)}%"
        else:
            for i in range(m):
                X[ix[i]+1][col*2+2] = args[i]
                X[ix[i]+1][col*2+3] = se_sgn[i]
            X[1+n][col*2+3] = round_sign_digits(self[key].LL, 5, 1)
            X[2+n][col*2+3] = self[key].df
            X[3+n][col*2+3] = f"{round(self[key].Rsqadj*100,1)}%"


class join_table_column:
    def __init__(self, stats, ll, panel):
        self.stats = stats
        self.LL = ll.LL
        self.df = panel.df
        self.args = ll.args
        self.Rsq, self.Rsqadj, self.LL_ratio, self.LL_ratio_OLS = stat.goodness_of_fit(
            ll, True, panel)
        self.instruments = panel.input.Z_names[1:]
        self.pqdkm = panel.pqdkm
        self.Y_name = panel.input.Y_names


def round_sign_digits(x, digits, min_digits=0):
    d = int(np.log10(abs(x)))
    return np.round(x, max((digits-1, d+min_digits))-d)
