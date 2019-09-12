from numpy import *
from minimizer import directions, minimizer, stepsize
from functions import rosenbrock

def main():
    x_init = array([
        [0.],
        [0.]
    ])
    f = rosenbrock.f
    grad_f = rosenbrock.grad_f
    hess_f = rosenbrock.hess_f

    direction_selection_params = {
        "grad_f" : grad_f
    }

    step_size_selection_params = {
        "f"         :   f,
        "grad_f"    :   grad_f,
        "alpha"     :   1e-1,
        "sigma"     :   1e-3,
        "beta"      :   1e-1
    }
    
    x, d, alpha, n_iters = minimizer.minimize(
        x_init = x_init, 
        f = f,
        grad_f = grad_f,
        epsilon = 1e-3,
        direction_selection_function = directions.steepest_descent,
        step_size_selection_function = stepsize.fixed,
        direction_selection_params = direction_selection_params,
        step_size_selection_params = step_size_selection_params 
    )
    print(x[-1])
    return

if __name__ == "__main__":
    main()