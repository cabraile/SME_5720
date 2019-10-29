from numpy import *
from numpy.linalg import *

"""
For every method in this module, the parameters are passed as dictionaries so the minimizer method for the sake of
modularization of the minimizer. 

For that to work, the dictionaries must follow the indicated parameters list, e.g., you can call the fixed step size
selection only setting a value for alpha (that can be the same for any other method). However, for using the Armijo
step size selection, you require to set f, grad_f, alpha, sigma and beta on the dictionary.

Global parameters dictionary
-------------------------
f: function
    The cost function to be minimized.
grad_f: function
    The gradient of the function to be minimized.
alpha: float
    The initial step size to be considered.
sigma: float
    "How much" the armijo condition restricts the step size.
beta: float
    The reduction of the step size.
"""

def fixed(x, params):
    return params["alpha"]

def armijo(x, params):
    f = params["f"]
    grad_f = params["grad_f"]
    alpha_k = params["alpha"]
    sigma = params["sigma"]
    beta = params["beta"]
    while( f(x - alpha_k * grad_f(x) ) > f(x) - alpha_k * sigma * norm( grad_f(x) ) ):
        alpha_k = alpha_k * beta
    return alpha_k

# TODO
def gll(x, params):
    return

# TODO
def zh(x, params):
    return