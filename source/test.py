from numpy import *
from numpy import *
from minimizer import directions, minimizer, stepsize
from functions import quadratic, rosenbrock

from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D

def plot_path(x,f,d):
    fig = figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = []
    ys = []
    zs = []
    us = []
    vs = []
    for k in range(len(x)):
        xs.append(x[k][0,0])
        ys.append(x[k][1,0])
        zs.append(f(x[k]))
        print(f(x[k]))
    for i in range(len(d)):
        us.append(d[i][0,0])
        vs.append(d[i][1,0])
    ax.plot(xs=xs, ys=ys, zs=zs, c="k")
    ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], c = 'b', label="Last x")
    ax.scatter([xs[0]], [ys[0]], [zs[0]], c="r" , label = 'Initial x')
    show()
    return

def main():
    x_init = array([
        [0.],
        [0.]
    ])
    f = quadratic.f
    grad_f = quadratic.grad_f
    hess_f = quadratic.hess_f

    direction_selection_params = {
        "grad_f" : grad_f,
        "hess_f" : hess_f
    }

    step_size_selection_params = {
        "f"         :   f,
        "grad_f"    :   grad_f,
        "alpha"     :   5e-2,
        "sigma"     :   0,
        "beta"      :   1e-2
    }
    
    x, d, alpha, n_iters = minimizer.minimize(
        x_init = x_init, 
        f = f,
        grad_f = grad_f,
        epsilon = 1e-3,
        direction_selection_function = directions.newton,
        step_size_selection_function = stepsize.armijo,
        direction_selection_params = direction_selection_params,
        step_size_selection_params = step_size_selection_params 
    )
    print("--------------")
    print("K = {}".format(n_iters))
    print(x[-1])
    plot_path(x,f,d)
    return

if __name__ == "__main__":
    main()