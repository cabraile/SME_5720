from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def plot_path(x,f, max_pts=500):
    """
    x: list of ndarray.
        All the points on the way to the result
    f: function.
        Function to be optimized
    """
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
    colors = []
    for idx in range(len(x)):
        color = cm.hot(1.0 - (zs[idx] - min_f )/ (max_f - min_f ) )
        colors.append(color)
    ax.scatter(xs=xs, ys=ys, zs=zs, color=colors)
    show()
    return
