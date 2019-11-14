from numpy import *
from minimizer import directions, minimizer, stepsize
from functions import quadratic, rosenbrock
from graphics import plotter

def minimize(f, grad_f, hess_f):
    x_init = array([
        [0.],
        [0.]
    ])

    direction_selection_params = {
        "grad_f" : grad_f,
        "hess_f" : hess_f
    }

    step_size_selection_params = {
        "f"         :   f,
        "grad_f"    :   grad_f,
        "alpha"     :   5e-3,
        "sigma"     :   0.5,
        "beta"      :   1e-1
    }

    x, d, alpha, n_iters = minimizer.minimize(
        x_init = x_init,
        f = f,
        grad_f = grad_f,
        epsilon = 1e-3,
        direction_selection_function = directions.newton,
        step_size_selection_function = stepsize.fixed,
        direction_selection_params = direction_selection_params,
        step_size_selection_params = step_size_selection_params,
        max_iters=1e3
    )
    return x, d, alpha, n_iters

def main():
    f = rosenbrock.f
    grad_f = rosenbrock.grad_f
    hess_f = rosenbrock.hess_f
    x, d, alpha, n_iters = minimize(f, grad_f, hess_f)
    print("--------------")
    print("K = {}".format(n_iters))
    print(x[-1])
    plot.plot_path(x,f)
    return

if __name__ == "__main__":
    main()
