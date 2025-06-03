import numpy as np 

class nPendulum:
    def __init__(self, m, l, lcom, I, n, m_dof):
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

def forward_dynamics_double_pendulum(x, u, model, g=9.81):
    l1,lcom1,m1,I1=model.l[0],model.lcom[0],model.m[0],model.I[0]
    l2,lcom2,m2,I2=model.l[1],model.lcom[1],model.m[1],model.I[1]
    q=x[0:2]
    v=x[2:4]
    m11 = I1 + I2 + l1**2*m2 + 2*l1*lcom2*m2*np.cos(q[1])  
    m22 = I2 
    m12 = I2 + l1*lcom2*m2*np.cos(q[1])
    m21 = I2 + l1*lcom2*m2*np.cos(q[1])     
    M = np.block([[m11, m12], [m21, m22]])  # inertia matrix   
    c1 = -2*l1*lcom2*m2*np.sin(q[1])*v[0]*v[1] - l1*lcom2*m2*np.sin(q[1])*v[1]**2
    c2 = l1*lcom2*m2*np.sin(q[1])*v[0]**2
    C = np.array([c1, c2])                 # coriolis + centrigual forces 
    g1 = g*lcom1*m1*np.sin(q[0]) + g*m2*(l1*np.sin(q[0]) + lcom2*np.sin(sum(q)))    
    g2 = g*m2*lcom2*np.sin(sum(q))
    G = np.array([g1, g2])                 # gravity vector
    θd = v
    ωd = np.linalg.solve(M, (u - C - G))
    return np.concatenate((θd, ωd))


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
        dynamics = lambda x,u: forward_dynamics_double_pendulum(x, u, model)        
        return rk4(x, u, h, dynamics)