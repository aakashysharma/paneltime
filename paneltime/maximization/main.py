#!/usr/bin/env python
# -*- coding: utf-8 -*-


from ..output import stat_functions as stat
from .. import system_settings
if system_settings.cython:
    from .. import likelihood_cython as logl
else:
    from .. import likelihood as logl

from ..output import communication as comm
from ..output import output
from . import init


import numpy as np
import time
import itertools
from queue import Queue
import os

EPS = 3.0e-16
TOLX = (4*EPS)
GTOL = 1e-5


TEST_ITER = 30


def maximize(panel, args, mp, t0, comm):

    task_name = 'maximization'

    if mp is None or panel.args.initial_user_defined:
        node = 5
        d = maximize_node(panel, args.args_v, 0.001, {}, {}, 0, False, False)
        d['node'] = node
        return d

    tasks = []
    a = get_directions(panel, args, mp.cpu_count)
    for i in range(len(a)):
        tasks.append(
            f'maximization.maximize_node(panel, {list(a[i])}, 0.000001, inbox, outbox, slave_id, False, True)\n'
        )
    evalnodes = EvaluateNodes(mp, len(tasks), t0, panel)
    mp.exec(tasks, task_name)

    while True:
        cb = mp.callback(task_name)
        if mp.callback_active:
            maxidx, bestix = evalnodes.get(cb[:len(tasks)])
        else:
            while sum(mp.check_state()) > 0:
                time.sleep(0.1)
            maxidx, bestix = get_final_res(mp, tasks, task_name)

        if not maxidx is None:
            break
        if not bestix is None:
            cb[bestix]['node'] = bestix
            comm.get(cb[bestix])

    cb_max = mp.collect(task_name, maxidx)
    cb_max['node'] = maxidx
    return cb_max


def get_cb_property(cb, kw, nonevalue=None, ndarray=True):
    values = [d[kw] if kw in d else nonevalue for d in cb]
    if ndarray:
        return np.array(values)
    return values


def get_final_res(mp, tasks, task_name):
    res = mp.collect(task_name)[:len(tasks)]
    f = get_cb_property(res, 'f', -1e+300)
    maxidx = f.index(max(f))
    return maxidx, maxidx


class EvaluateNodes:
    def __init__(self, mp, n, t0, panel):
        self.panel = panel
        self.mp = mp
        self.n_tests = 0
        self.n = n
        self.t = t0
        self.included = list(range(n))
        self.accuracy = self.panel.options.accuracy.value
        self.debug_mode = panel.options.debug_mode.value
        self.create_file()
        self.max_ll = None

    def get(self, cb):
        terminated = np.array(self.mp.check_state())[:self.n] == 0
        f = get_cb_property(cb, 'f')

        flist = np.array(f)
        flist = [i for i in f if not i is None]

        if np.all(terminated):
            ix = self.finish(cb, f)
            return ix, ix
        if len(flist) == 0:
            return None, None
        bestix = list(f).index(max(flist))
        return None, bestix

    def finish(self, cb, f):
        conv = get_cb_property(cb, 'conv', 0) > 0
        det = get_cb_property(cb, 'det', None, False)
        convdet = conv[0]*det[0] != 0
        if np.any(convdet):
            conv = convdet
        if np.any(conv):
            fval = max(f[conv])
        else:
            fval = max(f)
        ix = list(f).index(fval)
        self.write_to_file(cb, f)
        return ix

    def write_to_file(self, cb, f):
        "For debugging"
        if not self.debug_mode:
            return
        conv = get_cb_property(cb, 'conv', 0) > 0
        ci = get_cb_property(cb, 'CI', 1e+300)
        ci_anal = get_cb_property(cb, 'CI_anal', 1e+300)
        x = get_cb_property(cb, 'x', None, False)
        se = get_cb_property(cb, 'se', [None], False)
        rsq_st = get_cb_property(cb, 'rsq_st', [None], False)
        rsq = get_cb_property(cb, 'rsq_st', [None], False)
        var_RE = get_cb_property(cb, 'var_RE', [None], False)
        var_u = get_cb_property(cb, 'var_u', [None], False)
        det = get_cb_property(cb, 'det', None, False)

        txt = ''
        for i in range(len(f)):
            txt += ';'.join(
                str(s) for s in ([i, f[i], ci[i], ci_anal[i]] +
                                 list(x[i]) +
                                 list(se[i]) +
                                 list(rsq_st[i]) +
                                 list(rsq[i]) +
                                 [var_RE[i], var_u[i], conv[i]])
            ) + '\n'
            print(
                f"f:{f[i]}, x:{x[i][:3]},CI:{ci[i]}, CI_anal: {ci_anal[i]}, det: {det[i]}, conv:{conv[i]}")
        fl = open(self.data_file, 'a')
        fl.write(txt)
        fl.close()

    def create_file(self):
        "For debugging"
        if not self.debug_mode:
            return
        panel = self.panel
        self.data_file = "data.csv"
        gf_head = ['Rsq', 'Rsqadj', 'LL_ratio', 'LL_ratio_OLS']
        txt = ';'.join(['sid', 'll', 'CI', 'CI analytical'] +
                       panel.args.names_v +
                       ['se ' + i for i in panel.args.names_v] +
                       ['st ' + i for i in gf_head] +
                       gf_head +
                       ['var_RE', 'var_u', 'converged']) + '\n'
        exists = os.path.exists(self.data_file)
        fl = open(self.data_file, 'a')
        if not exists:
            fl.write(txt)
        fl.close()


def get_directions(panel, args, n):
    if n == 1:
        return [args.args_v]
    d = args.positions
    size = panel.options.initial_arima_garch_params.value
    pos = [d[k][0] for k in ['rho', 'lambda'] if len(d[k])]
    perm = np.array(list(itertools.product(
        [-1, 0, 1], repeat=len(pos))), dtype=float)
    perm[:, :2] = perm[:, :2]*0.1
    a = np.array([args.args_v for i in range(len(perm))])
    a[:, pos] = perm
    return a


def maximize_node(panel, args, gtol=1e-5, inbox={}, outbox={}, slave_id=0, nummerical=False, diag_hess=False):

    res, ll = init.maximize(args, inbox, outbox, panel,
                            gtol, TOLX, nummerical, diag_hess, slave_id)
    # debug from . import debug
    # debug.save_reg_data(ll, panel)

    H, G, g = res['H'], res['G'], res['g']

    ll.standardize(panel)
    res['rsq_st'] = stat.goodness_of_fit(ll, True, panel)
    res['rsq'] = stat.goodness_of_fit(ll, True, panel)
    res['var_RE'] = panel.var(ll.e_RE)
    res['var_u'] = panel.var(ll.u)
    return res


def run(panel, args, mp, window, exe_tab, console_output):
    t0 = time.time()
    comm = Comm(panel, args, mp, window, exe_tab, console_output, t0)
    summary = Summary(comm, panel, t0)

    return summary


class Summary:
    def __init__(self, comm, panel, t0):
        self.output = comm.channel.output

        # coefficient statistics:
        self.coef_params = comm.ll.args.args_v
        self.coef_names = comm.ll.args.caption_v
        self.coef_se, self.coef_se_robust = output.sandwich(
            comm.H, comm.G, comm.g, comm.constr, panel, 100)
        self.table = output.RegTableObj(
            panel, comm.ll, comm.g, comm.H, comm.G, comm.constr, comm.dx_norm, self.output.model_desc)
        self.coef_tstat = self.table.d['tstat']
        self.coef_tsign = self.table.d['tsign']
        self.coef_codes = self.table.d['sign_codes']
        self.coef_025 = self.table.d['conf_low']
        self.coef_0975 = self.table.d['conf_high']

        # other statistics:
        self.time = time.time() - t0
        self.panel = panel
        self.ll = comm.ll
        self.log_likelihood = comm.ll.LL

        self.converged = comm.conv
        self.hessian = comm.H
        self.gradient_vector = comm.g
        self.gradient_matrix = comm.G

        self.x = comm.x
        self.count_samp_size_orig = panel.orig_size
        self.count_samp_size_after_filter = panel.NT_before_loss
        self.count_deg_freedom = panel.df
        N, T, k = panel.X.shape
        self.count_ids = N
        self.count_dates = T

        self.statistics = output.Statistics(comm.ll, panel)
        self.CI, self.CI_n = self.output.get_CI(comm.constr)

        self.its = comm.its
        self.dx_norm = comm.dx_norm
        self.msg = comm.msg
        self.comm = comm
        self.t0 = t0

    def __str__(self, statistics=True, diagnostics=True, df_accounting=True):
        return self.comm.channel.print_final(self.comm, self.t0,  statistics, diagnostics, df_accounting)

    def results(self, return_string=False):
        t = self.table.table()[0]
        if return_string:
            return t
        print(t)
        return t

    def print_df_summary(self, return_string=False):
        t = self.statistics.gen_df_str(self.panel)
        if return_string:
            return t
        print(t)

    def print_model_summary(self, return_string=False):
        t = self.statistics.gen_mod_fit()
        if return_string:
            return t
        print(t)

    def print_adf_stat(self, return_string=False):
        t = self.statistics.adf_str()
        if return_string:
            return t
        print(t)

    def predict(self, signals=None):
        # debug:
        # self.ll.predict(self.panel.W_a[:,-2], self.panel.W_a[:,-1])
        N, T, k = self.panel.W_a.shape
        if signals is None:
            pred = self.ll.predict(self.panel.W_a[:, -1], None)
            return pred
        if not hasattr(signals, '__iter__'):  # assumed float
            signals = np.array([signals])
        else:
            signals = np.array(signals)
        if len(signals.shape) > 1 or signals.shape[0] != k-1:
            raise RuntimeError(
                "Signals must be a float or a one dimensional vector with the same size as variables assigned to HF argument")

        signals = np.append([1], signals)
        pred = self.ll.predict(self.panel.W_a[:, -1], signals.reshape((1, k)))
        return pred


class Comm:
    def __init__(self, panel, args, mp, window, exe_tab, console_output, t0):
        self.current_max = None
        self.mp = mp
        self.start_time = t0
        self.panel = panel
        self.channel = comm.get_channel(
            window, exe_tab, self.panel, console_output)
        d = maximize(panel, args, mp, t0, self)

        self.get(d)

    def get(self, d):
        if not 'g' in d:
            return False
        (self.f, self.its, self.incr, self.x, self.perc, self.task,
         self.dx_norm, self.dx, self.H, self.G, self.g, self.alam, self.rev,
         self.msg, self.conv, self.constr, terminate, self.node) = (
            d['f'], d['its'], d['incr'], d['x'], d['perc'], d['task'], d['dx_norm'], d['dx'],
            d['H'], d['G'], d['g'], d['alam'], d['rev'], d['msg'], d['conv'], d['constr'], d['terminate'], d['node'])

        self.ll = logl.LL(self.x, self.panel, self.constr)
        self.print_to_channel(self.msg, self.its, self.incr,
                              self.ll, self.perc, self.task, self.dx_norm)

    def print_to_channel(self, msg, its, incr, ll, perc, task, dx_norm):
        self.channel.set_output_obj(ll, self, dx_norm)
        self.channel.set_progress(perc, msg, task=task)
        self.channel.update(self, its, ll, incr, dx_norm)
        ev = np.abs(np.linalg.eigvals(self.H))**0.5
        try:
            det = np.linalg.det(self.H)
        except:
            det = 'NA'
        if (not self.panel.options.supress_output.value) and self.f != self.current_max:
            print(f"node: {self.node}, its: {self.its},  LL:{self.f}")
        self.current_max = self.f
