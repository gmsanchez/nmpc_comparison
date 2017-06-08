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
scripts = ["comparison_casadi.py",
           "comparison_casadi_colloc.py",
           "comparison_casadi_ltv_qp.py"
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

execfile("vdp_plot_all.py")
