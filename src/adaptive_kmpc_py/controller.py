from adaptive_kmpc_py.predictor import EDMD
import numpy as np
import scipy as sp
from scipy import sparse  
import osqp 

class adaptiveKMPC:
    def __init__(self,  edmd, Q: np.ndarray, Qf: np.ndarray, R: np.ndarray, r: np.ndarray, H, ul: np.ndarray, uu: np.ndarray) -> None:
        self.edmd = edmd
        self.p, self.m = edmd.get_dims()
        self.H = H 
        self.N = r.shape[1]
        
        # build weight matrices for the augmented dynamics
        Qf = np.diag(np.concatenate((Qf, np.zeros(self.m))))                
        Q = np.concatenate((Q, np.zeros(self.m)))
        self.Q_bold = np.diag(np.repeat(Q, H))
        self.Q_bold[-self.p-self.m:,-self.p-self.m:] = Qf 
        self.R_bold = np.diag(np.repeat(R, H))
        
        self.Psi_r = self.edmd.linear_model.observables.fit(r)        # required before evaluating the observables         
        self.Psi_r = self.edmd.linear_model.observables.transform(r).T

        self.ul_bold_const, self.uu_bold_const, self.C_Delta = self.build_input_constraints(ul, uu)
        self.C_Delta = sparse.csc_matrix(self.C_Delta)
        
        # initialize dummy QP (required by OSQP, since sparsity pattern is 'locked in' when calling OSQP setup function)  
        A = np.ones((self.p,self.p))
        B = np.ones((self.p,self.m))
        P,q = self.build_quadratic_cost(A, B, self.Psi_r, np.ones(self.p), np.ones(self.m))
        
        self.solver = osqp.OSQP()
        self.solver.setup(P=sparse.csc_matrix(P), 
                          q=q, 
                          A=sparse.csc_matrix(self.C_Delta), 
                          l=self.ul_bold_const,  
                          u=self.uu_bold_const, 
                          warm_starting=True,
                          verbose=False)
        
        self.u_prev = np.zeros(self.m)
        
    def update_buffer(self, x: np.ndarray, u: np.ndarray, t: np.ndarray) -> None:
        self.edmd.buffer.update_buffer(x, u, t)
        
    def build_quadratic_cost(self, A: np.ndarray, B: np.ndarray, Psi_r: np.ndarray, z0: np.ndarray, u_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        A_hat, B_hat = self.augment_model(A,B)
        A_bold, B_bold = self.build_prediction_matrices(A_hat, B_hat)
        
        z0_hat = np.concatenate((z0, u_prev))
        
        r_bold = np.zeros((self.p+self.m)*self.H)
        for i in range(self.H):
            istart = i*(self.p+self.m)
            r_bold[istart:istart+self.p] = Psi_r[:,i]
        
        P = B_bold.T @ self.Q_bold @ B_bold + self.R_bold
        q = 2*B_bold.T @ self.Q_bold @ (A_bold@z0_hat - r_bold)     
        return P,q 
         
    
    def build_input_constraints(self, ul: np.ndarray, uu: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ul_bold_const = np.kron(np.ones(self.H), ul)
        uu_bold_const = np.kron(np.ones(self.H), uu)
        C_Delta = np.kron(np.tri(self.H), np.identity(self.m))        
        return ul_bold_const, uu_bold_const, C_Delta
            
    def augment_model(self, A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        A_hat = np.block([[A,B],
                         [np.zeros((self.m,self.p)), np.identity(self.m)]])        
        B_hat = np.block([[B],
                         [np.identity(self.m)]])
        return A_hat, B_hat
    
    def build_prediction_matrices(self, A_hat: np.ndarray, B_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_aug,m_aug = B_hat.shape
        A_bold = np.zeros((self.H*n_aug, n_aug))
        B_bold = np.zeros((self.H*n_aug, self.H*m_aug))
        
        for i in range(self.H):
            A_bold[i*n_aug:(i+1)*n_aug, :n_aug]= np.linalg.matrix_power(A_hat, i+1)
            for j in range(self.H):
                if i>= j:
                    B_bold[i*n_aug:(i+1)*n_aug, j*m_aug:(j+1)*m_aug]= np.linalg.matrix_power(A_hat, i-j) @ B_hat
        return A_bold, B_bold
        
    def get_control(self, x0: np.ndarray, k: int):
        x0 = x0.reshape((1,x0.size))
        self.edmd.linear_model.observables.fit(x0)
        z0 =  self.edmd.linear_model.observables.transform(x0)

        A,B = self.edmd.fit()
        Psi_r = np.zeros((self.p, self.H))
        for i in range(k,k+self.H):
            if i <= self.N:
                Psi_r[:,i-k] = self.Psi_r[:,i]                     
            else: 
                Psi_r[:,i-k] = self.Psi_r[:,-1]                     
        P,q = self.build_quadratic_cost(A, B, Psi_r, np.squeeze(z0), self.u_prev)
        
        ul_bold = self.ul_bold_const - np.repeat(self.u_prev, self.H)
        uu_bold = self.uu_bold_const - np.repeat(self.u_prev, self.H)
        
        P = sparse.csc_matrix(P)
        self.solver.update(Px=sparse.triu(P).data, q = q, l = ul_bold, u = uu_bold, Ax = self.C_Delta.data)

        delta_u_bold = self.solver.solve().x
        u = self.u_prev + delta_u_bold[0:self.m]
        self.u_prev = u 
        
        return u 
    
    
        
