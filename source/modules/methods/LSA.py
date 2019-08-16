# -*- coding: utf-8 -*-

from numpy import *
from numpy.linalg import *

class LSA():
    """
    Linear Search Algorithm.

    Attributes
    --------------------------
    f: function.
        The function to be minimized.
    grad_f: function.
        The gradient of the function to be minimized.
    hess_f: function.
        The hessian of the function to be minimized.
    sigma: float.
        
    beta: float.
        Decay of the step on the gradient direction.
    mu: float.
        Step size that the next X is from the gradient of the current X.
    epsilon: float.
        Tolerance of the size of the gradient.
    _lambda:

    _theta:

    """
    def __init__(self):
        self.__clearParams__()
        return

    def __clearParams__(self):
        self.params = { 
            "f" : None,
            "grad_f": None,
            "hess_f": None,
            "epsilon": 1e-3,
            "mu": 1e-3,
            "sigma": 1e-2,
            "beta": 0.9,
            "_lambda": 0.1,
            "theta": 0.5
        }
        return

    def setParams(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value
        return

    def minimize(self, X_init, direction="adaptive"):
        """
        Params
        ----------------------
        X_init: ndarray.
            A column matrix of the initial guess for the input.
        direction: string.
            The direction the algorithm takes while searching for the minimizer. Possible 
            values (default is "adaptive"): 
            i) "adaptive": uses Newton's method for computing direction if the Hessian of X is 
            positive-definite. If not, chooses the oposite direction of the gradient.
            ii) "gradient": chooses the oposite direction of the gradient
            iii) "newton": uses Newton's method for computing the direction.
        """
        # Init params
        f = self.params["f"]
        grad_f = self.params["grad_f"]
        hess_f = self.params["hess_f"]
        epsilon = self.params["epsilon"]
        mu_init = self.params["mu"]
        sigma = self.params["sigma"]
        beta = self.params["beta"]
        _lambda = self.params["_lambda"]
        theta = self.params["theta"]
        # Iterations
        X = X_init
        steps = [X_init]
        while norm(grad_f(X)) > epsilon:
            # Init
            mu = mu_init
            d = None
            # Search direction definition
            if(direction == "adaptive"):
                # Newton direction
                d = - inv(hess_f(X)).dot(grad_f(X))
                # Check if direction is valid
                cond1 = transpose(d).dot(grad_f(X)) <= -theta * norm(d) * norm(grad_f(X))
                cond2 = norm(d) >= _lambda * norm(grad_f(X))
                # If not valid, changes direction
                if not ( cond1 and cond2 ):
                    # Uses the oposite direction of the gradient
                    d = - grad_f(X)
            elif (direction == "newton"):
                d = - inv(hess_f(X)).dot(grad_f(X))
            elif (direction == "gradient"):
                d = - grad_f(X)
            # Check Armijo condition
            armijo_cond = f( X + mu*d ) > f(X) + sigma * mu * transpose(d).dot(grad_f(X))
            while armijo_cond:
                mu = mu * beta
                armijo_cond = f( X + mu*d ) > f(X) + sigma * mu * transpose(d).dot(grad_f(X))
            # Update
            X = X + mu * d
            steps.append(X)
        return X, steps
