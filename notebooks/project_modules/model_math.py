from numpy import *
from project_modules import graph_utils

# AUXILIAR METHODS
# =====================================

def plus_operator(val):
    if val > 0:
        return val
    return 0.0

def O_ij(s_i, delta_s_i, s_j, delta_s_j):
    ret_val = 0
    if(s_i >= s_j):
        ret_val =  (1./(delta_s_j ** 4)) * plus_operator(delta_s_j ** 2 - (s_i - s_j) ** 2) ** 2
    else:
        ret_val =  (1./(delta_s_i ** 4)) * plus_operator(delta_s_i ** 2 - (s_i - s_j) ** 2) ** 2
    return ret_val

# =====================================

# ENERGY FUNCTIONS
# =====================================

def E_O(X, X_dims):
    sum_e = 0
    n = X.shape[0]
    for i in range(n):
        point_i = X[i]
        # Horizontal
        x_i = point_i[0]
        h_i = X_dims[i,0]
        # Vertical
        y_i = point_i[1]
        v_i = X_dims[i,1]
        for j in range(i+1,X.shape[0]):
            point_j = X[j]
            # Horizontal
            x_j = point_j[0]
            h_j = X_dims[j,0]
            overlapping_h = O_ij(x_i, h_i, x_j, h_j)
            # Vertical
            y_j = point_j[1]
            v_j = X_dims[j,1]
            overlapping_v = O_ij(y_i, v_i, y_j, v_j)
            # Sum
            sum_e += overlapping_h * overlapping_v
    sum_e = 2.0 / (n * (n+1) ) * sum_e
    return sum_e

def E_N(X, w, L, delta_x, delta_y):
    N = X.shape[0]
    norm_factor = (N ** 2.0)/( 2.0 * (linalg.norm(delta_x) ** 2 +linalg.norm(delta_y) ** 2) )
    diff_x = linalg.norm( L.dot(X[:,0]) - w * delta_x ) ** 2
    diff_y = linalg.norm(L.dot(X[:,1]) - w * delta_y) ** 2 
    diff_factor = ( diff_x + diff_y)
    sum_e = norm_factor * diff_factor
    return sum_e

# =====================================

# CONSTRAINTS
# =====================================

# r1: x_min-x_i <= 0 <=> x_i - x_min >= 0
def constr_fun_1(X, x_min):
    """
    X is the matrix (N,2) of points.
    """
    N = X.shape[0]
    constr = zeros((N))
    for i in range(N):
        constr[i] = X[i,0] - x_min
    return constr

# r2: x_i - x_max + h_i <= 0 <=> x_max - x_i - h_i >= 0
def constr_fun_2(X, x_max, X_dims):
    """
    X is the matrix (N,2) of points.
    """
    N = X.shape[0]
    constr = zeros((N))
    for i in range(N):
        constr[i] = x_max - X[i,0] - X_dims[i,0] - 1
    return constr

# r3: y_min-y_i <= 0 <=> y_i - y_min >= 0
def constr_fun_3(X, y_min):
    """
    X is the matrix (N,2) of points.
    """
    N = X.shape[0]
    constr = zeros((N))
    for i in range(N):
        constr[i] = X[i,1] -  y_min
    return constr

# r4: y_i - y_max + v_i <= 0 <=> y_max - y_i - v_i >= 0
def constr_fun_4(X, y_max, X_dims):
    """
    X is the matrix (N,2) of points.
    """
    N = X.shape[0]
    constr = zeros((N))
    for i in range(N):
        constr[i] = y_max - X[i,1]  - X_dims[i,1] - 1
    return constr

# =====================================


# DIFFERENCIATION
# =====================================

def del_O_del_vi(v_i, v_j, delta_v_i, delta_v_j):
    del_O = 0
    diff = (v_i - v_j)
    if v_i >= v_j:
        del_O = ( -4.0 / (delta_v_j ** 4) ) * diff * plus_operator( delta_v_j**2 - diff ** 2 )
    else:
        del_O = ( -4.0 / (delta_v_i ** 4) ) * diff * plus_operator( delta_v_i**2 - diff ** 2 )
    return del_O

def jac_E_O(X, X_dims):
    N = (X.shape[0])
    grad_E = empty((2*N))
    norm_factor = 2.0 / (N * (N+1))
    for k in range(N):
        grad_E_xk = norm_factor
        grad_E_yk = norm_factor
        x_k = X[k,0]; h_k = X_dims[k,0]
        y_k = X[k,1]; v_k = X_dims[k,1]
        for i in range(N):
            if(i == k):
                continue
            x_i = X[i,0]; h_i = X_dims[i,0]; 
            y_i = X[i,1]; v_i = X_dims[i,1]
            # Grad for x
            del_O_del_xk = del_O_del_vi(x_k, x_i, h_i, h_k)
            grad_E_xk += O_ij(y_i, v_i, y_k, v_k) * del_O_del_xk
            # Grad for y
            del_O_del_yk = del_O_del_vi(y_k, y_i, v_i, v_k)
            grad_E_yk += O_ij(x_i, h_i, x_k, h_k) * del_O_del_yk
        grad_E[k*2] = grad_E_xk
        grad_E[k*2+1] = grad_E_yk
    return grad_E

# =====================================