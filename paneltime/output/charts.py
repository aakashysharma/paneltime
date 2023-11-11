#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import stat_functions as stat
from . import stat_dist

import numpy as np
import os
try:
    from matplotlib import pyplot as plt
except:
    plt = None


class ProcessCharts():
    def __init__(self, panel):
        if plt is None:
            return
        self.panel = panel
        self.ll = None
        self.subplot = plt.subplots(1, figsize=(4, 2.5), dpi=75)
        self.chart_list = [
            ['histogram', self.histogram],
            ['correlogram', self.correlogram],
            ['correlogram_variance', self.correlogram_variance]
        ]

    def save_all(self, ll):
        self.ll = ll
        if self.ll is None:
            return
        for name, chart in self.chart_list:
            chart(f'img/{name}.png')

    def histogram(self, f):
        N, T, k = self.panel.X.shape
        fgr, axs = self.subplot
        n = self.ll.e_RE_norm_centered.shape[2]
        e = self.ll.e_RE_norm_centered[self.panel.included[2]].flatten()
        N = e.shape[0]
        e = e.reshape((N, 1))

        grid_range = 4
        grid_step = 0.05
        h, grid = histogram(e, grid_range, grid_step)
        norm = stat_dist.norm(grid, cdf=False)*grid_step

        axs.bar(grid, h, color='grey', width=0.025, label='histogram')
        axs.plot(grid, norm, 'green', label='normal distribution')
        axs.legend(prop={'size': 6})
        name = 'Histogram - frequency'
        axs.set_title(name)
        self.save(f)

    def correlogram(self, f):
        fgr, axs = self.subplot
        lags = 20
        rho = stat.correlogram(self.panel, self.ll.e_RE_norm_centered, lags)
        x = np.arange(lags+1)
        axs.bar(x, rho, color='grey', width=0.5, label='correlogram')
        name = 'Correlogram - residuals'
        axs.set_title(name)
        self.save(f)

    def correlogram_variance(self, f):
        N, T, k = self.panel.X.shape
        fgr, axs = self.subplot
        lags = 20
        e2 = self.ll.e_RE_norm_centered**2
        e2 = (e2-self.panel.mean(e2))*self.panel.included[3]
        rho = stat.correlogram(self.panel, e2, lags)
        x = np.arange(lags+1)
        axs.bar(x, rho, color='grey', width=0.5, label='correlogram')
        name = 'Correlogram - squared residuals'
        axs.set_title(name)
        self.save(f)

    def save(self, save_file):
        fgr, axs = self.subplot
        fgr.savefig(save_file)
        axs.clear()


def histogram(x, grid_range, grid_step):
    N, k = x.shape
    grid_n = int(2*grid_range/grid_step)
    grid = np.array(
        [i*grid_step-grid_range for i in range(grid_n)]).reshape((1, grid_n))
    ones = np.ones((N, 1))
    x_u = np.concatenate((ones, x >= grid), 1)
    x_l = np.concatenate((x < grid, ones), 1)
    grid = np.concatenate((grid.flatten(), [grid[0, -1]+grid_step]))
    histogram = np.sum((x_u*x_l), 0)
    if int(np.sum(histogram)) != N:
        raise RuntimeError('Error in histogram calculation')
    return histogram/N, grid
