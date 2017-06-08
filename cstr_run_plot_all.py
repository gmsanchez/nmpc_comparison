# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 17:28:17 2017

@author: guiss
"""
import time

methods = ["multiple_shooting",
           "collocation",
           "ltv_approximation"
           ]
scripts = ["cstr_comparison_casadi.py",
           "cstr_startup_colloc.py",
           "cstr_comparison_casadi_ltv.py"
           ]
Xdata = []
Udata = []
Tdata = []

for i in range(len(scripts)):
    print "Running script %d of %d: %s" % (i, len(scripts), scripts[i])
    print methods[i]
    # raw_input()
    execfile(scripts[i])
    Xdata.append(x)
    Udata.append(u)
    Tdata.append(ptimes)
    # Xdata[methods[i]] = x
    del x
    del u
    del ptimes
    time.sleep(0.1)

X = dict(zip(methods,Xdata))
U = dict(zip(methods,Udata))
T = dict(zip(methods,Tdata))

execfile("cstr_plot_all.py")
