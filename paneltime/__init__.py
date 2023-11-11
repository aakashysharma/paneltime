#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import likelihood_cython as logl
from . import parallel
from . import main
from . import options as opt_module
from . import info


import numpy as np
import os
import sys
import time
import pandas as pd

import inspect


mp = None

CALLBACK_ACTIVE = True


def enable_parallel():
    global mp
    N_NODES = 1
    PARALLEL = True  # change to false for debugging

    t0 = time.time()

    # temporary debug output is saved here:

    mp = parallel.Parallel(N_NODES, PARALLEL, CALLBACK_ACTIVE)

    mp.exec("from paneltime import maximization\n", 'init')

    print(f"parallel: {time.time()-t0}")


def execute(model_string, dataframe, ID=None, T=None, HF=None, instruments=None, console_output=True):
    """Maximizes the likelihood of an ARIMA/GARCH model with random/fixed effects (RE/FE)\n
          model_string: a string on the form 'Y ~ X1 + X2 + X3\n
          dataframe: a dataframe consisting of variables with the names usd in model_string, ID, T, HF and instruments\n
          ID: The group identifier\n
          T: the time identifier\n
          HF: list with names of heteroskedasticity factors (additional regressors in GARCH)\n
          instruments: list with names of instruments
          console_output: if True, GUI output is turned off (GUI output is experimental)
          """

    window = main.identify_global(
        inspect.stack()[1][0].f_globals, 'window', 'geometry')
    exe_tab = main.identify_global(
        inspect.stack()[1][0].f_globals, 'exe_tab', 'isrunning')

    r = main.execute(model_string, dataframe, ID, T, HF, options,
                     window, exe_tab, instruments, console_output, mp)

    return r


__version__ = info.version

options = opt_module.regression_options()
preferences = opt_module.application_preferences()
