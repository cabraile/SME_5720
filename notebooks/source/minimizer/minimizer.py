from numpy import *
from numpy.linalg import *

def minimize(
    x_init,                         # The initial point from which the minimizer will iterate
    f,                              # The function to be minimized
    grad_f,                         # The gradient of the cost function
    epsilon,                        # The minimum size the gradient has to be so the answer is accepted
    direction_selection_function,   # Function that chooses the direction for the next iteration
    step_size_selection_function,   # Function that evaluates the selected direction
    direction_selection_params,     # Parameters that will be passed to the direction selection function
    step_size_selection_params,     # Parameters that will be passed to the direction condition function
    max_iters = 1e4,
    callback = None                 # Callable that runs after each iteration. Provides x_k as parameter
):
    x = [ x_init ]
    d = [ ]
    alpha = [ ]

    k = 0
    x_k = x_init
    while ( norm(grad_f(x_k)) > epsilon and k < max_iters):
        # select direction
        d_k = direction_selection_function(x = x[k], params = direction_selection_params)

        # select step size
        alpha_k = step_size_selection_function(x = x[k], params = step_size_selection_params)

        # update
        x_k = x_k + alpha_k * d_k

        # finish iter
        x.append(x_k)
        d.append(d_k)
        alpha.append(alpha_k)
        k+=1
        if callback is not None :
            callback(x_k)
    return x, d, alpha, k
