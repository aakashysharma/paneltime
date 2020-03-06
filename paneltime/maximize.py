#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import stat_functions as stat
import loglikelihood as logl
import output
import sys
from tkinter import _tkinter




def lnsrch(ll, direction,mp):
	rmsg=''
	args=ll.args_v
	g = direction.g
	dx = direction.dx
	panel=direction.panel
	LL0=ll.LL
	constr=direction.constr
	
	
	if np.sum(g*dx)<0:
		dx=-dx
		rmsg="convex function at evaluation point, direction reversed - "	
	if mp is None:
		return lnsrch_single(args, dx,panel,constr,rmsg)
	else:
		return lnsrch_master(args, dx,panel,constr,mp,rmsg,LL0)
	

def lnsrch_single(args, dx,panel,constr,rmsg,f0=None):
	#not in use, will be removed
	d=dict()
	if f0 is None:
		f0=logl.LL(args, panel,constr)
	d[f0.LL]=[f0,0]
	m=0.5
	f05,f1=None,None
	for i in range(16+len(dx)):#Setting lmda so that the largest step is valid. Set ll.LL to return None when input is invalid
		lmda=1.1*m**i #Always try full Newton step first.
		if i>14:
			dx=dx*(np.abs(dx)<max(np.abs(dx)))
		x=f0.args_v+lmda*dx
		f=logl.LL(x,panel,constr)
		if not f.LL is None:
			if f1 is None:
				f1,l1=f,lmda
			elif f05 is None:
				f05,l05=f,lmda
				break
	if f1 is None or f05 is None:
		return f0,'no valid values within newton step in linesearch',0,False
	d[f1.LL],d[f05.LL]=[f1,l1],[f05,l05]
	lambda_pred=solve_square_func(f0.LL,0,f05.LL,l05,f1.LL,l1)
	if not lambda_pred is None:
		f_pred=logl.LL(f0.args_v+lambda_pred*dx,panel,constr) 
		if not f_pred.LL is None:	
			d[f_pred.LL]=[f_pred,lambda_pred]
		lmda=min((lambda_pred,lmda))
	f_max=max(d.keys())
	if f_max>f0.LL:#the function has  increased
		return d[f_max][0],rmsg + f"Linesearch success ({round(lmbda,4)} of Newton step)",d[f_max][1],False
	for j in range(1,12):
		s=(0.5**j)
		lm=lmda*s
		ll=logl.LL(f0.args_v+lm*dx,panel,constr) 
		if not ll.LL is None:
			if ll.LL>f0.LL:
				return ll, rmsg+f"Newton step {round(lmbda,4)} to big, found an increment at {lm} of Newton step",lm,True

	return f0,rmsg+'No increase in linesearch',0,False
		
		
def solve_square_func(f0,l0,f05,l05,f1,l1,default=None):
	try:
		b=-f0*(l05+l1)/((l0-l05)*(l0-l1))
		b-=f05*(l0+l1)/((l05-l0)*(l05-l1))
		b-=f1*(l05+l0)/((l1-l05)*(l1-l0))
		c=((f0-f1)/(l0-l1)) + ((f05-f1)/(l1-l05))
		c=c/(l0-l05)
		if c<0 and b>0:#concave increasing function
			return -b/(2*c)
		else:
			return default
	except:
		return default
	
	
def lnsrch_master(args, dx,panel,constr,mp,rmsg,LL0):
	mp.send_dict_by_file({'constr':constr})
	start=0
	end=1.5
	msg=''
	for i in range(4):
		delta=(end-start)/(mp.master.cpu_count-1)
		res=get_likelihoods(args, dx, panel, constr, mp,delta,start)
		if i==0:
			res0=res[0]		
		if (res[0,1]==0 and i==0) or (res[0,0]<=res0[0] and i>0) or np.isnan(res[0,0]):#best was no change
			start=delta/mp.master.cpu_count
			end=delta
		else:
			if i>0:
				msg=f'Found increment at {round(res[0,2],4)} of Newton step'
			break
	if i>0:
		res=np.append([res0],res,0)#ensuring the original is in the LL set
		srt=np.argsort(res[:,0])[::-1]
		res=res[srt]
	if LL0>res[0,0]:
		raise RuntimeWarning('Best linsearch is poorer than the starting point. You may have discovered a bug, please notify espen.sirnes@uit.no')
	try:
		lmda=solve_square_func(res[0,0], res[0,2],res[1,0], res[1,2],res[2,0], res[2,2],res[0,2])
		ll=logl.LL(args+lmda*dx,panel,constr)
		if ll.LL<res[0,0]:
			raise RuntimeError('Something wrong with ll. You may have discovered a bug, please notify espen.sirnes@uit.no')
	except:
		ll, lmda = mp.remote_recieve(f'f{res[0,1]}',res[0,1]), res[0,2]
	if lmda==0:
		return ll,rmsg+'No increase in linesearch',0,False
	if msg=='':
		msg=f"Linesearch success ({round(lmda,6)} of Newton step)"
	return ll,rmsg+msg,lmda,True

				
def get_likelihoods(args, dx,panel,constr,mp,delta,start):
	expr=[]	
	lamdas=[]
	args_lst=[]
	ids=range(mp.master.cpu_count)
	for i in ids:
		lmda=start+i*delta
		a=list(args+lmda*dx)
		lamdas.append(lmda)
		args_lst.append(a)
		expr.append([f"""
try:
	f{i}=lgl.LL({a}, panel,constr)
	LL{i}=f{i}.LL
except:
	LL{i}=None
""", f'LL{i}'])
	d=mp.remote_execute(expr)
	res=[]
	for i in ids:
		if not d[f'LL{i}'] is None:
			res.append([d[f'LL{i}'],i,lamdas[i]])
	if len(res)==0:
		return np.array([[np.nan, np.nan, np.nan]])
	res=np.array(res,dtype='object')
	srt=np.argsort(res[:,0])[::-1]
	res=res[srt]
	return res
	

	
	
def maximize(panel,direction,mp,args,tab):
	"""Maxmizes logl.LL"""

	its, convergence_limit	= 0, 0.01
	k, m, dx_norm,incr		= 0,     0,    None, 0
	H,  digits_precision    = None, 12
	msg,lmbda,newton_failed	='',	1, False
	direction.hessin_num, ll= None, None
	args_archive			= panel.input.args_archive
	ll=direction.init_ll(args)
	po=printout(tab,panel)
	if not printout_func(0.0,'Determining direction',ll,its,direction,incr,po):return ll,direction,po
	while 1:
		direction.get(ll,its,newton_failed)
		LL0=ll.LL
			
		#Convergence test:
		constr_conv=np.max(np.abs(direction.dx_norm))<convergence_limit
		unconstr_conv=np.max(np.abs(direction.dx_unconstr)) < convergence_limit 
		conv=constr_conv or (unconstr_conv and its>3)
		args_archive.save(ll.args_d,conv,panel.pqdkm)
		if conv: 
			printout_func(1.0,"Convergence on zero gradient; maximum identified",ll,its,direction,incr,po)
			return ll,direction,po

		if not printout_func(0.95,"Linesearch",ll,its,direction,incr,po):return ll,direction,po
		ll,msg,lmbda,ok=lnsrch(ll,direction,mp) 
		if not printout_func(1.0,msg,ll,its,direction,incr,po):return ll,direction,po


		incr=ll.LL-LL0

		its+=1
		
def printout_func(percent,msg,ll,its,direction,incr,po):
	po.printout(ll,its+1,direction,True,incr)	
	return direction.progress_bar(percent,msg)
		
	
def round_sign(x,n):
	"""rounds to n significant digits"""
	return round(x, -int(np.log10(abs(x)))+n-1)


def impose_OLS(ll,args_d,panel):
	beta,e=stat.OLS(panel,ll.X_st,ll.Y_st,return_e=True)
	args_d['omega'][0][0]=np.log(np.var(e*panel.included)*len(e[0])/np.sum(panel.included))
	args_d['beta'][:]=beta
	
class printout:
	def __init__(self,tab,panel,_print=True):
		self._print=_print
		self.tab=tab
		if tab is None:
			return
		self.panel=panel
		
	def printout(self,ll, its, direction, display_statistics,incr):
		if self.tab is None and self._print:
			print(ll.LL)
			return
		if not self._print:
			return
		#self.tab.ll=ll
		self.displaystats(display_statistics,ll)
				
		o=output.output(printout_format,ll, direction,self.panel.settings.robustcov_lags_statistics.value[1])
		o.add_heading(its,
					  top_header=" "*118+"constraints",
					  statistics=[['\nDependent: ',self.panel.input.y_name[0],None,"\n"],
								  ['Max condition index',direction.CI,3,'decimal']],
					  incr=incr)
		o.add_footer("Significance codes: .=0.1, *=0.05, **=0.01, ***=0.001\n"
					+'\n' 
					+ ll.err_msg)
		tab_stops=o.get_tab_stops(self.tab.box.text_box.config()['font'][4])
		self.tab.box.text_box.config(tabs=tab_stops)		
		self.print(o)
		self.prtstr=o.printstring
		
	def print(self,o):
		try:
			o.print(self.tab)
		except Exception as e:
			test1=e.args[0]=="main thread is not in main loop"
			#test2=e==_tkinter.TclError
			if test1:
				exit(0)
			else:
				raise e		
		
	def displaystats(self,display_statistics,ll):
		if display_statistics:
			process_charts=self.tab.charts
			process_charts.initialize(self.panel)
			process_charts.plot(ll)	



l=8
		#python variable name,	lenght,	  not string, display name,	  neg. values,	justification	next tab space
printout_format=[['names',		'namelen',	True,	'Variable names',			False,		'right', 		2],
				 ['args',		l,			False,	'Coef',						True,		'right', 		2],
				 ['dx_norm',	l,			False,	'direction',				True,		'right', 		2],
				 ['se_robust',	l,			False,	'SE(robust)',				True,		'right', 		3],
				 ['tstat',		l,			False,	't-stat.',					True,		'right', 		2],
				 ['tsign',		l,			False,	'sign.',					False,		'right', 		1],
				 ['sign_codes',	5,			True,	'',							False,		'left', 		2],
				 ['assco',		20,			True,	'collinear with',			False,		'center', 		2],
				 ['set_to',		6,			True,	'set to',					False,		'center', 		2],
				 ['cause',		50,			True,	'cause',					False,		'right', 		2]]			