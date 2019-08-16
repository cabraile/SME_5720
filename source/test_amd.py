# -*- coding: utf-8 -*-

#
from numpy import *

from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Implemented methods to be tested
from modules.functions import rosenbrock
from modules.methods.LSA import LSA


def main():
    testAMD()
    return

def plotFunction(x_list, y_list, z_list):
    fig = figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x_list, y_list, z_list)
    ax.scatter([x_list[-1] ], [y_list[-1]], [z_list[-1]], color="red" )
    show()
    return

def testAMD():
    X_init=array([[0.],[0.]])
    minimizer = LSA()
    minimizer.setParams(
        f = rosenbrock.f,
        grad_f = rosenbrock.grad_f,
        hess_f = rosenbrock.hess_f,
        epsilon = 1e-3,
        mu_init = 1e0,
        sigma = 1.0,
        beta = 0.9,
        _lambda = 0.2,
        theta = 0.5
    )
    X_solution, steps = minimizer.minimize(X_init = X_init, direction="gradient")
    x_list = [ X[0] for X in steps]
    y_list = [ X[1] for X in steps]
    f_x_y_list = [ rosenbrock.f( X ) for X in steps ]
    plotFunction(x_list, y_list, f_x_y_list)
    print("Took {} iters to finish.".format(len(f_x_y_list)))
    return


if __name__== "__main__":
    main()