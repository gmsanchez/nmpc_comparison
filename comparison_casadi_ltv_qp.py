# Control of the Van der Pol
# oscillator using pure casadi.
import casadi
import casadi.tools as ctools
import numpy as np
import matplotlib.pyplot as plt

import scipy.linalg
import time

import multiprocessing


#<<ENDCHUNK>>


def c2d(A, B, Delta, Bp=None, f=None, asdict=False):
    """
    Discretizes affine system (A, B, Bp, f) with timestep Delta.
    This includes disturbances and a potentially nonzero steady-state, although
    Bp and f can be omitted if they are not present.
    If asdict=True, return value will be a dictionary with entries A, B, Bp,
    and f. Otherwise, the return value will be a 4-element list [A, B, Bp, f]
    if Bp and f are provided, otherwise a 2-element list [A, B].
    """
    n = A.shape[0]
    I = np.eye(n)
    D = scipy.linalg.expm(Delta*np.vstack((np.hstack([A, I]),
                                           np.zeros((n, 2*n)))))
    Ad = D[:n, :n]
    Id = D[:n, n:]
    Bd = Id.dot(B)
    Bpd = None if Bp is None else Id.dot(Bp)
    fd = None if f is None else Id.dot(f)

    if asdict:
        retval = dict(A=Ad, B=Bd, Bp=Bpd, f=fd)
    elif Bp is None and f is None:
        retval = [Ad, Bd]
    else:
        retval = [Ad, Bd, Bpd, fd]
    return retval
    
def _calc_lin_disc_wrapper_for_mp_map(item):
    """ Function wrapper for map or multiprocessing.map . """
    _fi, _xi, _ui, _Delta = item
    Ai = _fi.jacobian(0, 0)(_xi, _ui)[0].full()
    Bi = _fi.jacobian(1, 0)(_xi, _ui)[0].full()
    # Gi = _fi.jacobian(2, 0)(_xi, _ui)[0].full()
    Ei = _fi(_xi, _ui).full().ravel() - Ai.dot(_xi).ravel() - Bi.dot(_ui).ravel() # - Gi.dot(_wi).ravel()
    # [Ai[:], Bi[:], Gi[:], Ei[:]] = c2d(Ai, Bi, _Delta, Gi, Ei)
    [Ai[:], Bi[:], _, Ei[:]] = c2d(Ai, Bi, _Delta, f=Ei)
    return Ai, Bi, Ei
    
def _calc_lin_disc_wrapper_for_mp_map_1(item):
    _fi, _xi, _ui, _Delta = item
    Ai, Bi, Ei = _fi(_xi,_ui)
    [Ai, Bi, _, Ei] = c2d(A=Ai, B=Bi, Delta=_Delta, f=Ei)
    return Ai, Bi, Ei
    
def _calc_lin_disc_wrapper_for_mp_map_2(item):
    Ai, Bi, Ei, _Delta = item
    [Ai[:,:], Bi[:,:], _, Ei[:,:]] = c2d(A=Ai, B=Bi, Delta=_Delta, f=Ei)
    return Ai, Bi, Ei

pool = multiprocessing.Pool()
# Define model and get simulator.
Delta = .25
Nt = 10
Nx = 2
Nu = 1
def ode(x,u,w=0):
    dxdt = [
        (1 - x[1]*x[1])*x[0] - x[1] + u,
        x[0]]
    return np.array(dxdt)

#<<ENDCHUNK>>

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

#<<ENDCHUNK>>

# Then get nonlinear casadi functions
# and rk4 discretization.
ode_casadi = casadi.Function(
    "ode",[x,u],[ode(x,u)])

jac_ode_casadi = casadi.Function("jac_ode",
                                 [x,u],
                                 [ode_casadi.jacobian(0,0)(x,u)[0],
                                  ode_casadi.jacobian(1,0)(x,u)[0],
                                  ode_casadi(x,u) - 
                                  casadi.mtimes([ode_casadi.jacobian(0,0)(x,u)[0],x]) -
                                  casadi.mtimes([ode_casadi.jacobian(1,0)(x,u)[0],u])])

jacMap = jac_ode_casadi.map("jacMap", "openmp", Nt)
#k1 = ode_casadi(x, u)
#k2 = ode_casadi(x + Delta/2*k1, u)
#k3 = ode_casadi(x + Delta/2*k2, u)
#k4 = ode_casadi(x + Delta*k3,u)
#xrk4 = x + Delta/6*(k1 + 2*k2 + 2*k3 + k4)    
#ode_rk4_casadi = casadi.Function(
#    "ode_rk4", [x,u], [xrk4])

#<<ENDCHUNK>>

# Define stage cost and terminal weight.
lfunc = (casadi.mtimes(x.T, x)
    + casadi.mtimes(u.T, u))
l = casadi.Function("l", [x,u], [lfunc])

Pffunc = casadi.mtimes(x.T, x)
Pf = casadi.Function("Pf", [x], [Pffunc])

#<<ENDCHUNK>>

# Bounds on u.
uub = 1
ulb = -.75

#<<ENDCHUNK>>

# Make optimizers.
x0 = np.array([0,1])
u0 = np.array([0])

c2d_keys = ['A', 'B', 'f']
c2d_vals = jac_ode_casadi(x0,u0)
c2d_dict = dict(zip(c2d_keys, c2d_vals))
c2d_dict["Delta"] = Delta

c2d(**c2d_dict)

# Create variables struct.
var = ctools.struct_symSX([(
    ctools.entry("x",shape=(Nx,),repeat=Nt+1),
    ctools.entry("u",shape=(Nu,),repeat=Nt),
)])
varlb = var(-np.inf)
varub = var(np.inf)
varguess = var(0)

# Create parameters struct.
par = ctools.struct_symSX([
    ctools.entry("Ad", repeat=Nt, shape=(Nx, Nx)),
    ctools.entry("Bd", repeat=Nt, shape=(Nx, Nu)),
    ctools.entry("fd", repeat=Nt, shape=(Nx, 1))])
parguess = par(0)

# Set initial values to parameters
[A0, B0, f0] = _calc_lin_disc_wrapper_for_mp_map([ode_casadi,x0,u0,Delta])
for t in range(Nt):
    parguess['Ad',t] = A0
    parguess['Bd',t] = B0
    parguess['fd',t] = f0
# Adjust the relevant constraints.
for t in range(Nt):
    varlb["u",t,:] = ulb
    varub["u",t,:] = uub

# Now build up constraints and objective.
obj = casadi.SX(0)
con = []
for t in range(Nt):
    con.append(var["x", t+1] -
        casadi.mtimes(par["Ad", t], var["x", t]) -
        casadi.mtimes(par["Bd", t], var["u", t]) -
        par["fd",t])                                 
    obj += l(var["x",t], var["u",t])
obj += Pf(var["x",Nt])

# Build solver object.
con = casadi.vertcat(*con)
conlb = np.zeros((Nx*Nt,))
conub = np.zeros((Nx*Nt,))

#nlp = dict(x=var, f=obj, g=con, p=par)
#nlpoptions = {
#    "ipopt.linear_solver" : "ma27",
#    "ipopt.print_level" : 0,
#    "ipopt.jac_c_constant" : "yes",
#    "ipopt.jac_d_constant" : "yes",
#    "ipopt.hessian_constant" : "yes",
#    "ipopt.max_cpu_time" : 60,
#    "ipopt" : {
#        "print_level" : 0,
#        "max_cpu_time" : 60,
#        "jac_c_constant" : "yes",
#        "jac_d_constant" : "yes",
#        "hessian_constant" : "yes",
#    },
#    "print_time" : False,
#    
#}
#solver = casadi.nlpsol("solver",
#    "ipopt", nlp, nlpoptions)

#<<ENDCHUNK>>

qpoptions = {"printLevel" : 'none', "print_time" : False, 'sparse':True}
qp = {'x':var, 'f':obj, 'g':con, 'p':par}
solver = casadi.qpsol('solver', 'qpoases', qp, qpoptions)

# Now simulate.
Nsim = 40
times = Delta*Nsim*np.linspace(0,1,Nsim+1)
x = np.zeros((Nsim+1,Nx))
x[0,:] = x0
u = np.zeros((Nsim,Nu))
ptimes = np.zeros((Nsim+1,1))
for t in range(Nsim):
    t0 = time.time()
    # Fix initial state.
    varlb["x",0,:] = x[t,:]
    varub["x",0,:] = x[t,:]
    varguess["x",0,:] = x[t,:]
    args = dict(x0=varguess,
                lbx=varlb,
                ubx=varub,
                lbg=conlb,
                ubg=conub,
                p=parguess)   
    
    #<<ENDCHUNK>>    
    
    # Solve nlp.    
    sol = solver(**args)
    status = "a" # solver.stats()["return_status"]
    optvar = var(sol["x"])

    outMap = jacMap(casadi.hcat(optvar['x',:-1]),casadi.hcat(optvar['u']))
    # _Ac = casadi.blocksplit(outMap[0],Nx,Nx)[0]
    # _Bc = casadi.blocksplit(outMap[1],Nx,Nu)[0]
    # _fc = casadi.blocksplit(outMap[2],Nx,1)[0]

    if t==0:    
        parguess["Ad",:], parguess["Bd",:], parguess["fd",:] = zip(*map(_calc_lin_disc_wrapper_for_mp_map_2,
                 zip(casadi.blocksplit(outMap[0],Nx,Nx)[0],
                     casadi.blocksplit(outMap[1],Nx,Nu)[0],
                     casadi.blocksplit(outMap[2],Nx,1)[0],
                     [Delta for _k in xrange(Nt)] )))
    else:
        parguess["Ad",0:-1] = parguess["Ad",1:]
        parguess["Bd",0:-1] = parguess["Bd",1:]
        parguess["fd",0:-1] = parguess["fd",1:]
        parguess["Ad",-1], parguess["Bd",-1], parguess["fd",-1] = _calc_lin_disc_wrapper_for_mp_map_1([jac_ode_casadi,optvar["x",-2],optvar["u",-1], Delta])
    
#    parguess["Ad",:], parguess["Bd",:], parguess["fd",:] = zip(*map(_calc_lin_disc_wrapper_for_mp_map_1,
#             zip([jac_ode_casadi for _k in xrange(Nt)],
#                  optvar["x",:-1],
#                  optvar["u"],
#                  [Delta for _k in xrange(Nt)] )))    
  

    #<<ENDCHUNK>>    
    t1 = time.time()
    # Print stats.
    print "%d: %s in %.4f seconds" % (t,status, t1 - t0)
    u[t,:] = optvar["u",0,:]
    ptimes[t] = t1-t0
    #<<ENDCHUNK>>

    # Simulate.
    vdpargs = dict(x0=x[t,:],
                   p=u[t,:])
    out = vdp(**vdpargs)
    x[t+1,:] = np.array(
        out["xf"]).flatten()

#<<ENDCHUNK>>
    
pool.close()
pool.join()


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