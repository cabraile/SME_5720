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
    delta_x = delta_x.reshape((N,1))
    delta_y = delta_y.reshape((N,1))
    vec_x = X[:,0].reshape((N,1))
    vec_y = X[:,1].reshape((N,1))
    norm_factor = (N ** 2.0)/( 2.0 * (linalg.norm(delta_x) ** 2 +linalg.norm(delta_y) ** 2) )
    diff_x = linalg.norm( L.dot(vec_x) - w * delta_x ) ** 2
    diff_y = linalg.norm(L.dot(vec_y) - w * delta_y) ** 2 
    diff_factor = ( diff_x + diff_y )
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
    del_O = 0.0
    diff = (v_i - v_j)
    if v_i >= v_j:
        factor = delta_v_j ** 2.0 - diff ** 2.0
        if(factor > 0):
            del_O = ( -4.0 / (delta_v_j ** 4.0) ) * diff * factor
    else:
        factor = delta_v_i ** 2.0 - diff ** 2.0
        if(factor > 0):
            del_O = ( -4.0 / (delta_v_i ** 4.0) ) * diff * factor
    return del_O

def jac_E_O(X, X_dims):
    N = (X.shape[0])
    jac_E = empty((2*N+1))
    norm_factor = 2.0 / (N * (N+1))
    for k in range(N):
        jac_E_xk = norm_factor
        jac_E_yk = norm_factor
        x_k = X[k,0]; h_k = X_dims[k,0]
        y_k = X[k,1]; v_k = X_dims[k,1]
        for i in range(N):
            if(i == k):
                continue
            x_i = X[i,0]; h_i = X_dims[i,0] 
            y_i = X[i,1]; v_i = X_dims[i,1]
            # Grad for x
            del_O_del_xk = del_O_del_vi(x_k, x_i, h_i, h_k)
            jac_E_xk += O_ij(y_i, v_i, y_k, v_k) * del_O_del_xk
            # Grad for y
            del_O_del_yk = del_O_del_vi(y_k, y_i, v_i, v_k)
            jac_E_yk += O_ij(x_i, h_i, x_k, h_k) * del_O_del_yk
        jac_E[k*2] = jac_E_xk
        jac_E[k*2+1] = jac_E_yk
    jac_E[-1] = 0.0
    return jac_E

def jac_E_N(vec_x, vec_y, w, L, delta_x, delta_y):
    N = vec_x.shape[0]
    jac_E = zeros((2*N+1))
    eta = ((1.0 * N)**2) / (2.0 * ( (linalg.norm(delta_x) ** 2 ) + (linalg.norm(delta_y) ** 2) ) )

    # Makes matrix multiplication easier, so does my life
    vec_x.reshape((N,1)) 
    vec_y.reshape((N,1))

    # Reduces computational burden
    L_dot_vec_x = L.dot(vec_x); L_dot_vec_y = L.dot(vec_y)
    w_times_delta_x = w * delta_x; w_times_delta_y = w * delta_y

    # partial w.r.t. x_k and y_k
    for k in range(N):
        sum_xk = 0.0
        sum_yk = 0.0
        for i in range(N):
            sum_xk += L[i,k] * ( L_dot_vec_x[i] - w_times_delta_x[i] )
            sum_yk += L[i,k] * ( L_dot_vec_y[i] - w_times_delta_y[i] )
        partial_x_k = 2.0 * eta * sum_xk
        partial_y_k = 2.0 * eta * sum_yk
        jac_E[2*k] = partial_x_k
        jac_E[2*k + 1] = partial_y_k

    # partial w.r.t. w
    x_side = 0
    y_side = 0
    for i in range(N):
        x_side += (L_dot_vec_x[i] - w_times_delta_x[i]) * delta_x[i]
        y_side += (L_dot_vec_y[i] - w_times_delta_y[i]) * delta_y[i]
    jac_E[-1] = - 2.0 * eta * (x_side + y_side)
    return jac_E
"""
def hess_E_N(L, delta_x, delta_y):
    N = L.shape[0]
    H = zeros((N*2+1,N*2+1))
    eta = ((1.0 * N)**2) / (2.0 * ( (linalg.norm(delta_x) ** 2 ) + (linalg.norm(delta_y) ** 2) ) )
    # (d/dx | d/dx) e (d/dy | d/dy)
    for i in range(N):
        for k in range(N):
            sum_factor = 0
            for j in range(N):
                sum_factor += L[j,k] * L[j,i]
            H[i*2,k*2] = 2.0 * eta * sum_factor
            H[i*2,k*2] = 2.0 * eta * sum_factor
    # Partials wrt w
    for k in range(N):
        sum_x = 0; sum_y = 0
        for j in range(N):
            sum_x += L[j,k] * delta_x[j]
            sum_y += L[j,k] * delta_y[j]
        H[-1,2*k]   = - 2.0 * eta * sum_x
        H[-1,2*k+1] = - 2.0 * eta * sum_y
    H[-1,-1] = N ** 2.0 # partial w.r.t. w
    return H

def d_o_v_i_partial_v_j(vec_v, vec_delta, i, j):
    d_o_v = 0.0
    if vec_v[i] >= vec_v[j] :
        f1 = vec_delta[j] ** 2 - (vec_v[i] - vec_v[j]) ** 2
        if f1 > 0.0: 
            d_o_v = -4.0 / (vec_delta[j] ** 4) * ( - plus_operator(f1) + 2.0 * (vec_v[i] - vec_v[j]) ** 2 )
    else: # x_i < x_j
        f2 = vec_delta[i] ** 2 - (vec_v[i] - vec_v[j]) ** 2
        if f2 > 0:
            d_o_v = -4.0 / (vec_delta[i] ** 4) * ( - plus_operator(f2) + 2.0 * (vec_v[i] - vec_v[j]) ** 2 )
    return d_o_v

def hess_E_O(vec_x, vec_y, vec_h, vec_v):
    N = vec_x.shape[0]
    H = zeros((N*2+1,N*2+1))
    norm_factor = 2.0 / (N * (N+1))
    for i in range(N):
        for j in range(N):
            if(i == j):
                continue

            # partial of d_x w.r.t. x
            o_y = O_ij(vec_y[i], vec_v[i], vec_y[j], vec_v[j])
            d_o_x = d_o_v_i_partial_v_j(vec_x, vec_h, i, j)
            H[i*2,j*2] =  norm_factor * o_y * d_o_x

            # partial of d_y w.r.t. y
            o_x = O_ij(vec_x[i], vec_h[i], vec_x[j], vec_h[j])
            d_o_y = d_o_v_i_partial_v_j(vec_y, vec_v, i, j)
            H[i*2+1,j*2+1] =  norm_factor * o_x * d_o_y

            # mixed partials
            dO_dx = del_O_del_vi(vec_x[i], vec_x[j], vec_h[i], vec_h[j])
            dO_dy = del_O_del_vi(vec_y[i], vec_y[j], vec_v[i], vec_v[j])
            H[i*2+1,j*2] = norm_factor * dO_dx * dO_dy
            H[i*2,j*2+1] = norm_factor * dO_dy * dO_dx

    # Partials w.r.t. w are 0
    return H
"""
# =====================================