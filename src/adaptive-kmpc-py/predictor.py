import pykoopman as pk
import numpy as np

def build_observables_dummy():

    observables = [lambda x: np.sin(x), lambda x: np.cos(x)]
    observable_names = [
        lambda s: f"sin{s}",
        lambda s: f"cos{s}",
    ]

    obs = pk.observables.CustomObservables(observables, observable_names=observable_names)
    return obs 

