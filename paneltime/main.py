#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Todo:


# capture singular matrix with test_small.csv
# make sure error in h function triggers an exeption

from .processing import panel
from .processing import model_parser
from . import maximization


import sys
import numpy as np
import warnings
import os
import time
import pandas as pd

N_NODES = 1


np.set_printoptions(suppress=True)
np.set_printoptions(precision=8)


def execute(model_string, dataframe, IDs_name, time_name, heteroscedasticity_factors, options, window,
            exe_tab, instruments, console_output, mp):
    """optimizes LL using the optimization procedure in the maximize module"""
    dataframe = dataframe.copy()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        if not exe_tab is None:
            if exe_tab.isrunning == False:
                return
        datainput = input_class(dataframe, model_string, IDs_name,
                                time_name, options, heteroscedasticity_factors, instruments)
        summary = run(datainput, options, mp, options.pqdkm.value,
                      window, exe_tab, console_output)

    return summary


class input_class:
    def __init__(self, dataframe, model_string, IDs_name, time_name, options, heteroscedasticity_factors, instruments):

        model_parser.get_variables(self, dataframe, model_string, IDs_name,
                                   time_name, heteroscedasticity_factors, instruments, options)
        self.descr = model_string
        self.n_nodes = N_NODES
        self.args = None
        if options.arguments.value != "":
            self.args = options.arguments.value


def run(datainput, options, mp, pqdkm, window, exe_tab, console_output):
    if not options.supress_output:
        print("Creating panel")
    pnl = panel.panel(datainput, options, pqdkm)

    if not mp is None:
        mp.send_dict({'panel': pnl})
        mp.collect('init')
        s = mp.dict_file.replace('\\', '/')
        mp.exec("panel.init()\n", 'panelinit')
    pnl.init()
    if not mp is None:
        mp.collect('panelinit')

    if not options.parallel.value:
        mp = None
    summary = maximization.run(
        pnl, pnl.args.args_init, mp, window, exe_tab, console_output)

    return summary


def indentify_dataset(glob, source):
    try:
        window = glob['window']
        datasets = window.right_tabs.data_tree.datasets
        for i in datasets:
            data_source = ' '.join(datasets[i].source.split())
            editor_source = ' '.join(source.split())
            if data_source == editor_source:
                return datasets[i]
    except:
        return False


def identify_global(globals, name, attr):
    try:
        variable = globals[name]
    except:
        variable = None
    if not hasattr(variable, attr):
        return None
    return variable
