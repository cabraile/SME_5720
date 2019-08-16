# -*- coding: utf-8 -*-
from numpy import *

def hess_f(X, a=1, b=100):
    x = X[0,0]
    y = X[1,0]
    d2f_dxdx = 2 - 4*b*(y-x**2) + 4 * b * (x ** 2 )
    d2f_dxdy = d2f_dydx = - 4 * b * x
    d2f_dydy = 2 * b
    return array([ [d2f_dxdx, d2f_dydx],[d2f_dxdy, d2f_dydy] ] )

def grad_f(X, a=1., b=100.):
    x = X[0,0]
    y = X[1,0]
    df_dx = -2.*(a - x)-b*4*(y-x**2)*x
    df_dy = 2 * b * (y - x ** 2)
    return array([[df_dx], [df_dy]])

def f(X, a=1.,b=100.):
    x = X[0,0]
    y = X[1,0]
    return (a-x)**2+b * (y-x**2)**2