# %% 
from adaptive_kmpc_py.controller import adaptiveKMPC
from adaptive_kmpc_py.predictor import EDMD
import matplotlib.pyplot as plt
import pykoopman as pk 
import numpy as np 

class nPendulum:
    m: float
    l: float
    lcom: float
    I: float
    n: int
    m_dof: int 
    def __init__(self, m: float,l: float,lcom: float, I: float, n: int, m_dof: int) -> None:
        self.m=m
        self.l=l
        self.lcom=lcom
        self.I=I
        self.n=n
        self.m_dof= m_dof          
        
def forward_dynamics_single_pendulum(x, u, model, g=9.81):
    theta1_dot = x[1]
    omega1_dot = 1/model.I * (u - model.m*g*model.lcom*np.sin(x[0]))
    return np.concatenate(([theta1_dot],omega1_dot))

def rk4(x,u,h,dynamics):
    k1 = dynamics(x, u)
    k2 = dynamics(x + 0.5*h*k1, u)
    k3 = dynamics(x + 0.5*h*k2, u)
    k4 = dynamics(x + h*k3, u)
    return x + h/6*(k1 + 2*k2 + 2*k3 + k4)    

def simulate(x, u, h, model):
    if model.n==2:
        dynamics = lambda x,u: forward_dynamics_single_pendulum(x, u, model)        
        return rk4(x, u, h, dynamics)
    elif model.n==4: 
        # TODO: Implement double pendulum dynamics 
        dynamics = lambda x,u: forward_dynamics_single_pendulum(x, u, model)        
        return rk4(x, u, h, dynamics)

# build single pendulum
m1 = 0.27                 # link 1 mass 
l1 = 0.4                  # link 1 length 
lc = 0.131                # distance rotational axis to center of mass (COM) of link 1
I1 = 0.007479             # moment of inertia of link 1 expressed in COM frame 
m_dof = 1;                  # number of DOF
n = 2
sp = nPendulum(m1,l1,lc,I1,n,m_dof)

observables = [lambda x1,x2: np.sin(x1), lambda x1,x2: np.cos(x1), lambda x1,x2: x2*np.sin(x1), lambda x1,x2: x2*np.cos(x1)]
observable_names = [
    lambda s,t: f"sin{s}",
    lambda s,t: f"cos{s}",
    lambda s,t: f"{t}*sin{s}",
    lambda s,t: f"{t}*cos{s}",
]
obs = pk.observables.CustomObservables(observables, observable_names=observable_names)

Nb = 500
edmd = EDMD(sp.n, sp.m_dof, Nb, obs)

# Controller parameters 
Q = np.concatenate(([1],np.zeros(5)))   # weights on tracking error (with these weights emphasis is put only on tracking of the joint angle θ1)
Qf =np.concatenate(([1],np.zeros(5)))   # weights on tracking error, final state 
R = np.ones(m_dof)*1                    # weights on control effort           
H = 30                                  # prediction horizon 
ul = np.array([-6.0])                   # lower limit on controls 
uu = np.array([6.0])                    # upper limit on controls 

r = np.loadtxt("trajectories/Xref.txt")
T_ref = np.loadtxt("trajectories/Tref.txt")

ctrl = adaptiveKMPC(edmd, Q, Qf, R, r, H, ul, uu)

# Preceding experiment for gathering data 
h = T_ref[2]-T_ref[1]               # discretization step length of the reference trajectory determines control freq. 
X_p = np.zeros((sp.n, Nb))                                  
U_p = np.zeros((m_dof, Nb-1))     
T_p = np.linspace(0,h*(Nb-1),Nb-1)

x0 = np.zeros(2)                          # initial state 
X_p[:,1] = x0
for i in range(Nb-1):
    U_p[:,i] = 0.15*np.cos(2*np.pi*0.05*i*h)           
    X_p[:,i+1] = simulate(X_p[:,i], U_p[:,i], h, sp)

# print(f"X: {X_p.shape}, U: {U_p.shape}, T: {T_p.shape}")

# fill circular buffer 
ctrl.update_buffer(X_p, U_p, T_p)

# Reference tracking 
N = 500                                                                     # duration of the tracking process is given by h*(N-1) (when the reference trajectory ends, the controller will track the last state)
X_a = np.zeros((sp.n, N))
U_a = np.zeros((m_dof, N-1))    
T_a = np.linspace(0,h*(N-1),N)

X_a[:,1] = X_p[:,-1]                                                        # assumption: the tracking process starts immediately after preceding experiment 
for i in range(N-1):
    print(f"iteration: {i+1}/ {N-1}")         
    U_a[:,i] = ctrl.get_control(X_a[:,i], i)                                # this function carries out the EDMD, building and solving of the MPC problem. Variable names inside the function are chosen according to the notation in the paper. 
    X_a[:,i+1] = simulate(X_a[:,i], U_a[:,i], h, sp)                        # apply control 
    ctrl.update_buffer(X_a[:,i+1].reshape((n,1)), U_a[:,i].reshape((m_dof,1)), np.array([T_a[i]]))                        # buffer is updated in every time step to provide new data for the EDMD 

# %% 
plt.plot(T_a[0:-1], U_a[0,:])
plt.show()

#%% 
plt.plot(T_a[0:], X_a[0,:])
plt.show()