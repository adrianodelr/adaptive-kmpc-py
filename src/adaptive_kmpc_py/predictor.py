import pykoopman as pk
import numpy as np
import collections      # https://stackoverflow.com/questions/4151320/efficient-circular-buffer

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
        assert u.shape[0] == self.n_inputs, f"u needs to be of dimension {self.n_states} x N"         
        assert len(t.shape) == 1, "time measurements need to be one dimensional" 
        for i in range(t.shape[0]):
            self.t.append(t[i])                
            for j in range(self.n_states):
                self.X[j].append(x[j,i])            
            for j in range(self.n_inputs):
                self.U[j].append(u[j,i])            

    def get_X(self) -> np.ndarray:
        return np.array(list(self.X))
    
    def get_U(self) -> np.ndarray:
        return np.array(list(self.U))
                
    # def get_X(self) -> list[collections.deque[float]]:
    #     return list(self.X)
    
    # def get_U(self) -> list[collections.deque[float]]:
    #     return list(self.U)

class EDMD:
    def __init__(self, n_states: int, n_inputs: int, N: int, observables: pk.observables.CustomObservables) -> None:
        self.linear_model = pk.Koopman(observables=observables, regressor=pk.regression.EDMDc())
        self.buffer = DataRingBuffer(n_states, n_inputs, N)
        self.p = len(observables.observables)-1+n_states            # identity is included by default as first observable  
        self.m = n_inputs
        
    def fit(self) -> tuple[np.ndarray, np.ndarray]:
        self.linear_model.fit(x=self.buffer.get_X().T, u=self.buffer.get_U().T)
        return np.array(self.linear_model.A),np.array(self.linear_model.B) 
        

    def get_dims(self) -> tuple[int, int]:
        return self.p,self.m
    
def build_observables_dummy():
    
    observables = [lambda x: np.sin(x), lambda x: np.cos(x)]
    observable_names = [
        lambda s: f"sin{s}",
        lambda s: f"cos{s}",
    ]

    obs = pk.observables.CustomObservables(observables, observable_names=observable_names)
    return obs 

