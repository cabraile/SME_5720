from numpy import *
from numpy import *
from minimizer import directions, minimizer, stepsize
from functions import quadratic, rosenbrock

from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def plot_path(x,f,d, max_pts=500):
    fig = figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = []
    ys = []
    zs = []
    for k in range(len(x)):
        xs.append(x[k][0,0])
        ys.append(x[k][1,0])
        zs.append( f(x[k]))
    max_f = 1.0 * max(zs)
    min_f = 1.0 * min(zs)
    step = len(x)/max_pts
    colors = []
    for idx in range(len(x)):
        #idx = i * step
        color = cm.hot(1.0 - (zs[idx] - min_f )/ (max_f - min_f ) )
        colors.append(color)
        #ax.scatter(xs=[xs[idx]], ys=[ys[idx]], zs=[zs[idx]], color=color)
    ax.scatter(xs=xs, ys=ys, zs=zs, color=colors)
    #ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], c = 'b', label="Last x")
    #ax.scatter([xs[0]], [ys[0]], [zs[0]], c="r" , label = 'Initial x')
    show()
    return

def main():
    x_init = array([
        [0.],
        [0.]
    ])
    f = rosenbrock.f
    grad_f = rosenbrock.grad_f
    hess_f = rosenbrock.hess_f

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
    print("--------------")
    print("K = {}".format(n_iters))
    print(x[-1])
    plot_path(x,f,d)
    return

if __name__ == "__main__":
    main()
