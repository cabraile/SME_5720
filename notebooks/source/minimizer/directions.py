from numpy import *
from numpy.linalg import *

"""
For every method in this module, the parameters are passed as dictionaries so the minimizer method for the sake of
modularization of the minimizer. 

For that to work, the dictionaries must follow the indicated parameters list, e.g., you can call the steepest descent
method and only set the gradient function (that can be the same for any other method). However, for using the Newton's
method, you require to set grad_f and hess_f on the dictionary.

Global parameters dictionary
-------------------------
f: function
    The cost function to be minimized.
grad_f: function
    The gradient of the function to be minimized.
hess_f: function
    The hessian of the input function
"""

def steepest_descent(x, params):
    return  - params["grad_f"](x)

def newton(x, params):
    return - inv(params["hess_f"](x)).dot(params["grad_f"](x))

def conjugate_gradients(x, params):
    return 

def quasi_newton(x, params):
    return