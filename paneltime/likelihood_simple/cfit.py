#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import numpy as np
import numpy.ctypeslib as npct
import ctypes as ct
p = os.path.join(Path(__file__).parent.absolute(), 'cfunctions')
if os.name == 'nt':
    cfit = npct.load_library('cfit.dll', p)
else:
    cfit = npct.load_library('cfit.so', p)

CDPT = ct.POINTER(ct.c_double)
CIPT = ct.POINTER(ct.c_uint)

n = 5
x = np.random.rand(n, n)
y = np.zeros((n, n))


def main():

    cfit.fit(x.ctypes.data_as(CIPT), y.ctypes.data_as(CIPT), n)
    print(y)


main()
