# Control of the Van der Pol
# oscillator using pure casadi.
import casadi
import casadi.tools as ctools
import numpy as np
import matplotlib.pyplot as plt

import time

# Define model and get simulator.
Delta = .25
Nt = 20
Nx = 3
Nu = 2
eps = 1e-6 # Use this as a small number.

T0 = 350
c0 = 1
r = .219
k0 = 7.2e10
E = 8750
U = 54.94
rho = 1000
Cp = .239
dH = -5e4

def ode(x,u):
    # Grab the states, controls, and disturbance. We would like to write
    #    
    # [c, T, h] = x[0:Nx]
    # [Tc, F] = u[0:Nu]
    # [F0] = d[0:Nd]
    #    
    # but this doesn't work in Casadi 3.0. So, we're stuck with the following:
    c = x[0]
    T = x[1]
    h = x[2]
    Tc = u[0]
    F = u[1]
    F0 = 0.1

    # Now create the ODE.
    rate = k0*c*np.exp(-E/T)
        
    dxdt = np.array([
        F0*(c0 - c)/(np.pi*r**2*h) - rate,
        F0*(T0 - T)/(np.pi*r**2*h)
            - dH/(rho*Cp)*rate
            + 2*U/(r*rho*Cp)*(Tc - T),    
        (F0 - F)/(np.pi*r**2)
    ])
    return dxdt

# Define symbolic variables.
x = casadi.SX.sym("x",Nx)
u = casadi.SX.sym("u",Nu)

# Make integrator object.
ode_integrator = dict(x=x,p=u,
    ode=ode(x,u))
intoptions = {
    "abstol" : 1e-8,
    "reltol" : 1e-8,
    "tf" : Delta,
}
vdp = casadi.integrator("int_ode",
    "cvodes", ode_integrator, intoptions)

# Then get nonlinear casadi functions
# and rk4 discretization.
ode_casadi = casadi.Function(
    "ode",[x,u],[ode(x,u)])

k1 = ode_casadi(x, u)
k2 = ode_casadi(x + Delta/2*k1, u)
k3 = ode_casadi(x + Delta/2*k2, u)
k4 = ode_casadi(x + Delta*k3,u)
xrk4 = x + Delta/6*(k1 + 2*k2 + 2*k3 + k4)    
ode_rk4_casadi = casadi.Function(
    "ode_rk4", [x,u], [xrk4])


# Steady-state values.
cs = .878
Ts = 324.5
hs = .659
Fs = .1
Tcs = 300
F0s = .1

xs = np.array([0.878, 324.5, 0.659])
us = np.array([300,   .1])

Q = .5*np.diag(xs**-2)
R = 2*np.diag(us**-2)
Qn = 10*Q # np.diag([1.0, 1.0, 1.0])
# Define stage cost and terminal weight.
lfunc = (casadi.mtimes([x.T, Q, x])
    + casadi.mtimes([u.T, R, u]))
l = casadi.Function("l", [x,u], [lfunc])

Pffunc = casadi.mtimes([x.T, Qn, x])
Pf = casadi.Function("Pf", [x], [Pffunc])

#<<ENDCHUNK>>

xlb = np.array([-np.inf, -np.inf, -np.inf])
xub = np.array([np.inf, np.inf, np.inf])

#<<ENDCHUNK>>

# Make optimizers.
x0 = np.array([.05*cs,.75*Ts,.5*hs])
u0 = np.array([Tcs, Fs])
umax = np.array([.05*Tcs,.15*Fs])
dumax = .2*umax

# Create variables struct.
var = ctools.struct_symSX([(
    ctools.entry("x",shape=(Nx,),repeat=Nt+1),
    ctools.entry("u",shape=(Nu,),repeat=Nt),
    ctools.entry("Du",shape=(Nu,),repeat=Nt),
)])
varlb = var(-np.inf)
varub = var(np.inf)
varguess = var(0)

par = ctools.struct_symSX([(
    ctools.entry("x_sp",shape=(Nx,),repeat=Nt+1),
    ctools.entry("u_sp",shape=(Nu,),repeat=Nt),
    ctools.entry("uprev", shape=(Nu, 1)),
    )])
parguess = par(0)

# Adjust the relevant constraints.
for t in range(Nt):
    varlb["u",t,:] = us-umax
    varub["u",t,:] = us+umax
    varlb["Du",t,:] = -dumax
    varub["Du",t,:] = +dumax
    varlb["x",t,:] = xlb
    varub["x",t,:] = xub
varlb["x",Nt,:] = xlb
varub["x",Nt,:] = xub

# Now build up constraints and objective.
obj = casadi.SX(0)

state_constraints = []
for t in range(Nt):
    state_constraints.append(ode_rk4_casadi(var["x",t],
        var["u",t]) - var["x",t+1])
    obj += l(var["x",t]-par["x_sp",t], var["u",t]-par["u_sp",t])
obj += Pf(var["x",Nt]-par["x_sp",Nt])

delta_constraints = []
if u0 is not None:
    delta_constraints.append(var["Du",0] - var["u", 0] + par["uprev"])
    for t in range(1, Nt-1):
        delta_constraints.append(var["Du",t] - var["u", t] + var["u", t-1])
            
# Build solver object.
con = []
con = state_constraints + delta_constraints
con = casadi.vertcat(*con)
conlb = np.zeros((Nx*Nt+Nu*(Nt-1),))
conub = np.zeros((Nx*Nt+Nu*(Nt-1),))

nlp = dict(x=var, f=obj, g=con, p=par)
nlpoptions = {
    "ipopt" : {
        "print_level" : 0,
        "max_cpu_time" : 60,
    },
    # "ipopt.linear_solver" : "ma27",
    "ipopt.print_level" : 0,
    "ipopt.max_cpu_time" : 60,
    "print_time" : False,
    
}
solver = casadi.nlpsol("solver",
    "ipopt", nlp, nlpoptions)

#<<ENDCHUNK>>

varguess["x",0,:] = xs
for i in range(1,Nt+1):
    vdpargs = dict(x0=np.array(varguess["x",i-1,:]).flatten(),
                   p=us)
    out = vdp(**vdpargs)
    varguess["x",i,:] = np.array(
        out["xf"]).flatten()
    

# Now simulate.
Nsim = 100
times = Delta*Nsim*np.linspace(0,1,Nsim+1)
x = np.zeros((Nsim+1,Nx))
x[0,:] = x0
# varguess["u",:,:] = np.array([300, 0.1])
parguess["x_sp",:,:] = xs #np.array([0.878, 324.5, 0.659])
parguess["u_sp",:,:] = us #np.array([300, 0.1])
parguess["uprev",:,:] = us
u = np.zeros((Nsim,Nu))
ptimes = np.zeros((Nsim+1,1))
for t in range(Nsim):
    t0 = time.time()
    # Fix initial state.    
    varlb["x",0,:] = x[t,:]
    varub["x",0,:] = x[t,:]
    varguess["x",0,:] = x[t,:]
    args = dict(x0=varguess,
                p=parguess,
                lbx=varlb,
                ubx=varub,
                lbg=conlb,
                ubg=conub)   
    
    #<<ENDCHUNK>>    
    
    # Solve nlp.    
    sol = solver(**args)
    status = solver.stats()["return_status"]
    optvar = var(sol["x"])
    
    #<<ENDCHUNK>>    
    
    t1 = time.time()
    # Print stats.
    print "%d: %s in %.4f seconds" % (t,status, t1 - t0)
    u[t,:] = np.array(
        optvar["u",0,:]).flatten()
    parguess["uprev",:,:] = u[t,:]
    ptimes[t] = t1-t0

    
    #<<ENDCHUNK>>    
    
    # Simulate.
    vdpargs = dict(x0=x[t,:],
                   p=u[t,:])
    out = vdp(**vdpargs)
    x[t+1,:] = np.array(
        out["xf"]).flatten()

#<<ENDCHUNK>>
    
# Plots.
fig = plt.figure()
numrows = max(Nx,Nu)
numcols = 2

# u plots. Need to repeat last element
# for stairstep plot.
u = np.concatenate((u,u[-1:,:]))
for i in range(Nu):
    ax = fig.add_subplot(numrows,
        numcols,numcols*(i+1))
    ax.step(times,u[:,i],"-k")
    ax.set_xlabel("Time")
    ax.set_ylabel("Control %d" % (i + 1))

# x plots.    
for i in range(Nx):
    ax = fig.add_subplot(numrows,
        numcols,numcols*(i+1) - 1)
    ax.plot(times,x[:,i],"-k",label="System")
    ax.set_xlabel("Time")
    ax.set_ylabel("State %d" % (i + 1))

fig.tight_layout(pad=.5)
fig.show()
