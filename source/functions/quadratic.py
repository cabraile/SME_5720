from numpy import *

# f(x) = x'Qx-bx
__QUADRATIC_Q = array([
    [1., 0.,],
    [0., 1. ]
])
__QUADRATIC_b = array([
    [0.],
    [0.]
])

def hess_f(X):
    Q = __QUADRATIC_Q
    return Q

def grad_f(X):
    Q = __QUADRATIC_Q
    b = __QUADRATIC_b
    return Q.dot(X) - b

def f(X):
    Q = __QUADRATIC_Q
    b = __QUADRATIC_b
    return (transpose(X).dot(Q.dot(X)) - transpose(b).dot(X))[0,0]
