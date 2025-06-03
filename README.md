# adaptive-kmpc-py
This is a Python implementation of the **adaptive Koopman model predictive control** (KMPC) algorithm, proposed in [https://arxiv.org/abs/2503.17902](https://arxiv.org/abs/2503.17902) (currently under review).  

An equivalent Julia implementation can be found in the [official paper repository](https://github.com/adrianodelr/adaptive-koopman-mpc).


### Adaptive KMPC algorithm
The controller allows approximating nonlinear system dynamics from experimental data as a linear system, which is then embedded in a convex model predictive controller. This eliminates the need for determining a dynamic model via first principles. Additionally, the model is updated online from the closed-loop feedback, which enables control of systems with varying dynamic parameters. 

The algorithm is explained in depth in the linked paper, and verified with real-system control experiments on a 1R and 2R robot system. To ensure understanding of the code, variable names are chosen according to the notation used in the paper.  

The following process schematic may help in getting a quick overview of the algorithm and the code.  
<img src="docs/adaptive_KMPC_scheme.png" alt="adaptive_KMPC_scheme" width="1000"/>


- For initial model build the controller relies on availability of data that partly capture the underlying system dynamics. These data can be obtained by perturbing the system with a sequence of 
random controls and recording the system response, which is denoted as `preceding experiment`. 
- Data are then used to fill a `circular buffer`, operatin under *first-in-first-out* logic. 
- Subsequently, Extended Dynamic Mode Decomposition is carried out on the buffer data (for more detailed explanation you can consult the paper), leading to a `linear model` in the discrete time domain.
- Casting the linear model in a `convex quadratic program` along with some reference trajectory, constraints, and feedback on the current state allows solving for an optimal control input that is applied to the system. 
- The system response, along with the applied control are fed back into the circular buffer, such that the model adapts to the current operating point.  


