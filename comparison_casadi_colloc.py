# Control of the Van der Pol
# oscillator using pure casadi.
import casadi
import casadi.tools as ctools
import numpy as np
import matplotlib.pyplot as plt

import time
#<<ENDCHUNK>>

# Degree of interpolating polynomial
d = 3

# Get collocation points
tau_root = [0]+casadi.collocation_points(d, 'legendre')

# Coefficients of the collocation equation
C = np.zeros((d+1,d+1))

# Coefficients of the continuity equation
D = np.zeros(d+1)

# Coefficients of the quadrature function
B = np.zeros(d+1)

# Construct polynomial basis
for j in range(d+1):
    # Construct Lagrange polynomials to get the polynomial basis at the collocation point
    p = np.poly1d([1])
    for r in range(d+1):
        if r != j:
            p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j]-tau_root[r])

    # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
    D[j] = p(1.0)

    # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
    pder = np.polyder(p)
    for r in range(d+1):
        C[j,r] = pder(tau_root[r])

    # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
    pint = np.polyint(p)
    B[j] = pint(1.0)


# Define model and get simulator.
Delta = .5
Nt = 20
Nx = 2
Nu = 1
# Number of collocation points
Nc = d
def ode(x,u):
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
ode_casadi = casadi.Function(
    "ode",[x,u],[ode(x,u)])
#<<ENDCHUNK>>

# Define stage cost and terminal weight.
lfunc = (casadi.mtimes(x.T, x)
    + casadi.mtimes(u.T, u))
l = casadi.Function("l", [x,u], [lfunc])

Pffunc = casadi.mtimes(x.T, x)
Pf = casadi.Function("Pf", [x], [Pffunc])

#<<ENDCHUNK>>

# Bounds on u.
uub = 1.0
ulb = -.75

#<<ENDCHUNK>>

# Make optimizers.
x0 = np.array([0,1])

# Create variables struct.
var = ctools.struct_symSX([(
    ctools.entry("x",shape=(Nx,),repeat=Nt+1),
    ctools.entry("u",shape=(Nu,),repeat=Nt),
    ctools.entry("xc", shape=(Nx,Nc), repeat=Nt),
)])
varlb = var(-np.inf)
varub = var(np.inf)
varguess = var(0)

# Adjust the relevant constraints.
for t in range(Nt):
    varlb["u",t,:] = ulb
    varub["u",t,:] = uub

# Now build up constraints and objective.
obj = casadi.SX(0)
#con = []
for t in range(Nt):
#    con.append(ode_rk4_casadi(var["x",t],
#        var["u",t]) - var["x",t+1])
    obj += l(var["x",t], var["u",t])
obj += Pf(var["x",Nt])


con = []
# For all finite elements



for k in range(Nt):
    colloc_args = []
    colloc_args += [var['x'][k]]
    colloc_args += casadi.blocksplit(var['xc'][k],Nx,1)[0]
    # colloc_args += [var['x'][k+1]]
    # For all collocation points
    for j in range(1,Nc+1):
        
        # Get an expression for the state derivative at the collocation point
        xp_jk = 0
        for r in range (d+1):
          # xp_jk += C[r,j]*var["xc",k][:,r]
            xp_jk += C[r,j]*colloc_args[r]
      
        # Add collocation equations to the NLP
        fk = ode_casadi(colloc_args[j], var["u",k])
        con.append(Delta*fk - xp_jk)
        
        # print [Delta*fk - xp_jk]
        # raw_input()
        #lbg.append(NP.zeros(nx)) # equality constraints
        #ubg.append(NP.zeros(nx)) # equality constraints

    # Get an expression for the state at the end of the finite element
    xf_k = 0
    for r in range(d+1):
          xf_k += casadi.mtimes(D[r],colloc_args[r])

    # Add continuity equation to NLP
    con.append(var['x'][k+1] - xf_k)
    #lbg.append(NP.zeros(nx))
    #ubg.append(NP.zeros(nx))
  
# Concatenate constraints
#g = vertcat(*g)


# Build solver object.
con = casadi.vertcat(*con)
conlb = np.zeros((con.shape[0],))
conub = np.zeros((con.shape[0],))

nlp = dict(x=var, f=obj, g=con)
nlpoptions = {
    "ipopt" : {
        "print_level" : 0,
        "max_cpu_time" : 60,
    },
    "ipopt.linear_solver" : "ma27",
    "ipopt.print_level" : 0,
    "ipopt.max_cpu_time" : 60,
    "print_time" : False,
    
}
solver = casadi.nlpsol("solver",
    "ipopt", nlp, nlpoptions)

#<<ENDCHUNK>>

# Now simulate.
Nsim = 40
times = Delta*Nsim*np.linspace(0,1,Nsim+1)
x = np.zeros((Nsim+1,Nx))
x[0,:] = x0
u = np.zeros((Nsim,Nu))
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
    u[t,:] = optvar["u",0,:]
    
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