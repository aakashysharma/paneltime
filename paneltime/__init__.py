#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(__file__.replace("__init__.py",''))
import time
import multi_core as mc



DEBUG_MODE = True
	
N_NODES = 10

t0=time.time()

path = os.getcwd()
if not DEBUG_MODE:
	mp = mc.Master(1, path)
	mp.exec(["import maximize\n"
			"import multi_core as mc\n"
			f"mp = mc.Master({N_NODES},'{path}/mp')\n" 
			"mp.exec('import loglikelihood as logl\\n'\n"
			"'import maximize', 'init')\n"], 'init')
	mp_debug = None
else:
	mp = None
	mp_debug = mc.Master(N_NODES, path)
	mp_debug.exec("import loglikelihood as logl\n"
				  "import maximize", 
				  'init')
	
print(f"mc: {time.time()-t0}")


from matplotlib import pyplot  as plt #matplotlib will not load late

import main
import options as opt_module
import inspect
import numpy as np
import loaddata
import pandas as pd


def execute(model_string,dataframe, ID=None,T=None,HF=None,instruments=None, console_output=False):
	"""optimizes LL using the optimization procedure in the maximize module"""
	window=main.identify_global(inspect.stack()[1][0].f_globals,'window')
	exe_tab=main.identify_global(inspect.stack()[1][0].f_globals,'exe_tab')
	r=main.execute(model_string,dataframe,ID, T,HF,options,window,exe_tab,instruments, console_output, mp, mp_debug)
	return r

def load_json(fname):

	if False:#detects previously loaded dataset in the environment
		dataframe=main.indentify_dataset(globals(),fname)
		if (not dataframe==False) and (not dataframe is None):
			return dataframe	
	try:
		dataframe=main.loaddata.load_json(fname)
	except FileNotFoundError:
		raise RuntimeError("File %s not found" %(fname))
	return dataframe


def load(fname,sep=None):

	"""Loads data from file <fname>, asuming column separator <sep>.\n
	Returns a dataframe (a dictionary of numpy column matrices).\n
	If sep is not supplied, the method will attemt to find it."""
	if False:#detects previously loaded dataset in the environment
		dataframe=main.indentify_dataset(globals(),fname)
		if (not dataframe==False) and (not dataframe is None):
			return dataframe	
	try:
		dataframe=main.loaddata.load(fname,sep)
	except FileNotFoundError:
		raise RuntimeError("File %s not found" %(fname))
	return dataframe

def load_SQL(conn,sql_string):

	"""Loads data from an SQL server, using sql_string as query"""
	if False:#detects previously loaded dataset in the environment
		dataframe=main.indentify_dataset(globals(),sql_string)
		if (not dataframe==False) and (not dataframe is None):
			return dataframe
	dataframe=main.loaddata.load_SQL(sql_string,conn)
	#except RuntimeError as e:
	#	raise RuntimeError(e)
	return dataframe
		
	
options=opt_module.regression_options()
preferences=opt_module.application_preferences()

