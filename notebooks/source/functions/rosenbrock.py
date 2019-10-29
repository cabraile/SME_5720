from numpy import *

_ROSENBROCK_PARAM_a = 1
_ROSENBROCK_PARAM_b = 100

def hess_f(X):
    a = _ROSENBROCK_PARAM_a
    b = _ROSENBROCK_PARAM_b
    del_2_f = empty((2,2))
    del_2_f[0,0] = - 4 * b * (X[1,0] - X[0,0]**2) + 8 * b * X[0,0]**2 + 2
    del_2_f[0,1] = - 4 * b * X[0,0]
    del_2_f[1,0] = - 4 * b * X[0,0]
    del_2_f[1,1] = 2 * b
    return del_2_f

def grad_f(X):
    a = _ROSENBROCK_PARAM_a
    b = _ROSENBROCK_PARAM_b
    del_f = empty((2,1))
    del_f[0,0] = - 2 * a + 4 * b * X[0,0] ** 3 - 4 * b * X[0,0] * X[1,0] + 2 * X[0,0]
    del_f[1,0] = 2 * b * (X[1,0] - X[0,0] ** 2)
    return del_f

def f(X):
    a = _ROSENBROCK_PARAM_a
    b = _ROSENBROCK_PARAM_b
    return ((a - X[0,0]) ** 2 + b * (X[1,0] - X[0,0] ** 2) ** 2)
