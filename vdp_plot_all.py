# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 18:23:45 2017

@author: guiss
"""
import matplotlib.pyplot as plt
plt.close("all")

Nx = 2
Nu = 1

linestyle = ['-','-','-','--','--','--.', ':', ':']
markers = ["D","s","^","v","D",">"]

x_lbl = ["Multiple Shooting", "Collocation", "LTV Approximation"]
                  
xlabelsize = 18
ylabelsize = 14
                  
for v in range(Nx):
    fig = plt.figure()
    for i in range(len(methods)):
        ax = fig.add_subplot(1,1,1)
        ax.get_yaxis().get_major_formatter().set_scientific(False)
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
            
        ax.plot(times,X[methods[i]][:,v], label=x_lbl[i],
                alpha=0.5, markevery=1, marker=markers[i],
                linestyle=linestyle[i], linewidth=1.5)
        ax.grid()
        ax.legend()
        
        ax.set_ylabel(r"$x_%d$"%(v+1),fontsize=xlabelsize)
        ax.set_xlabel(r"Time [s]",fontsize=ylabelsize) 
        
    pltScale = 0.1
    (minlim,maxlim) = ax.get_xlim()
    offset = .05*pltScale*(maxlim - minlim)
    ax.set_xlim(minlim - offset, maxlim + offset)
    (minlim,maxlim) = ax.get_ylim()
    offset = .5*pltScale*(maxlim - minlim)
    ax.set_ylim(minlim - offset, maxlim + offset)
    plt.tight_layout()
    plt.draw()
    
    plt.savefig("vdp_x_%d.pdf"%(v+1),format='PDF')
        

for v in range(Nu):
    fig = plt.figure()
    for i in range(len(methods)):
        ax = fig.add_subplot(1,1,1)
        ax.get_yaxis().get_major_formatter().set_scientific(False)
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
            
        ax.step(times,U[methods[i]][:,v], label=x_lbl[i],
                alpha=0.5, markevery=1, marker=markers[i],
                where='post',linewidth=1.5)
        ax.grid()
        ax.legend()
        
        ax.set_ylabel(r"$u_%d$"%(v+1),fontsize=xlabelsize)
        ax.set_xlabel(r"Time [s]",fontsize=ylabelsize) 
        
    pltScale = 0.1
    (minlim,maxlim) = ax.get_xlim()
    offset = .05*pltScale*(maxlim - minlim)
    ax.set_xlim(minlim - offset, maxlim + offset)
    (minlim,maxlim) = ax.get_ylim()
    offset = .5*pltScale*(maxlim - minlim)
    ax.set_ylim(minlim - offset, maxlim + offset)
    plt.tight_layout()
    plt.draw()

    plt.savefig("vdp_u_%d.pdf"%(v+1),format='PDF')

fig = plt.figure()
for i in range(len(methods)):
    ax = fig.add_subplot(1,1,1)
    # ax.get_yaxis().get_major_formatter().set_scientific(False)
    # ax.get_yaxis().get_major_formatter().set_useOffset(False)
        
    ax.plot(times[:-1],T[methods[i]][:-1,v], label=x_lbl[i],
            alpha=0.5, markevery=1, marker=markers[i],
            linestyle=linestyle[i], linewidth=1.5)
    ax.grid()
    ax.legend()
    
    ax.set_ylabel(r"Execution time [s]",fontsize=14)
    ax.set_xlabel(r"Time [s]",fontsize=ylabelsize) 
    print "Mean process time for %s: %.5f"%(methods[i], np.mean(T[methods[i]][:-1]))
    
pltScale = 0.05
(minlim,maxlim) = ax.get_xlim()
offset = .5*pltScale*(maxlim - minlim)
ax.set_xlim(minlim - offset, maxlim + offset)
#    (minlim,maxlim) = ax.get_ylim()
#    offset = .5*pltScale*(maxlim - minlim)
#    ax.set_ylim(minlim - offset, maxlim + offset)
plt.tight_layout()
plt.draw()

plt.savefig("vdp_proc_time.pdf",format='PDF')
    
