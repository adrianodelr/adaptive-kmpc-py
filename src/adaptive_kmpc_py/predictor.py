import pykoopman as pk
import numpy as np
from numba import njit
import collections      

class DataRingBuffer:
    def __init__(self, n_states: int, n_inputs: int, N: int) -> None:
        self.X = [collections.deque(maxlen=N) for _ in range(n_states)]
        self.U = [collections.deque(maxlen=N-1) for _ in range(n_inputs)]
        self.t =  collections.deque(maxlen=N)
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.N = N 
    
    def update_buffer(self, x: np.ndarray, u: np.ndarray, t: np.ndarray) -> None:
        
        assert x.shape[0] == self.n_states, f"x needs to be of dimension {self.n_states} x N"
        assert u.shape[0] == self.n_inputs, f"u needs to be of dimension {self.n_states} x N-1"         
        assert len(t.shape) == 1, "time measurements need to be one dimensional" 
        
        for i in range(t.shape[0]):
            self.t.append(t[i])                
            for j in range(self.n_states):
                self.X[j].append(x[j,i])
        for i in range(u.shape[1]):            
            for j in range(self.n_inputs):
                self.U[j].append(u[j,i])
                  
    def get_X(self) -> np.ndarray:
        return np.array([list(x) for x in self.X])
    
    def get_U(self) -> np.ndarray:
        return np.array([list(u) for u in self.U])


class EDMD:
    def __init__(self, n_states: int, n_inputs: int, N: int, observables: pk.observables.CustomObservables) -> None:
        self.linear_model = pk.Koopman(observables=observables, regressor=pk.regression.EDMDc())
        self.buffer = DataRingBuffer(n_states, n_inputs, N)
        self.p = len(observables.observables)-1+n_states            # identity is included by default as first observable  
        self.m = n_inputs
 
    def fit(self) -> tuple[np.ndarray, np.ndarray]:
        X = self.buffer.get_X().T
        U = self.buffer.get_U()
        Z = self.linear_model.observables.transform(X).T
        Z_plus = Z[:,1:]
        Z = Z[:,:-1]        
        Omega = np.vstack((Z,U))        
        K, _, _, _ = np.linalg.lstsq(Omega.T, Z_plus.T, rcond=None)
        K = K.T
        return K[:,0:self.p],K[:,self.p:] 
        
    def get_dims(self) -> tuple[int, int]:
        return self.p,self.m
    
