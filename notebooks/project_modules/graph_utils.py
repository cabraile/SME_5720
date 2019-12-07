from numpy import *
from matplotlib.pyplot import *
from matplotlib.collections import PatchCollection

def _check_connected(i, G, V, adjacency_list):
    N = G.shape[0]
    if V[i] == 0: # not visited
        adjacency_list.append(i)
        V[i] = 1
        for idx in range(N):
            if G[i,idx] != 0: # is connected
                _check_connected(idx, G, V, adjacency_list) 
    return

def components_distance(X, c1, c2):
    """
    Compute the minimum distance between two components.
    Parameters
    ----------------
    X: Nx2 array. 
        The set of x,y points.
    c1,c2: list.
        List of indices of unconnected points in the adjacency graph
    """
    min_dist = inf
    min_i = -1
    min_j = -1
    for i in c1:
        p_i = X[i]
        for j in c2:
            p_j = X[j]
            dist = linalg.norm(p_i - p_j)
            if min_dist > dist:
                min_dist = dist
                min_i = i
                min_j = j
    return {"dist" : min_dist, "c1_idx" : min_i, "c2_idx" : min_j}

def get_unconnected_components(G):
    """
    Returns a list of unconnected components. Each component is a 
    list of the connected vertices within it.
    """
    N = G.shape[0]
    V = zeros((N)) # Visited elements
    unconnected_components = []
    for i in range(N):
        adjacency_list = []
        _check_connected(i, G, V, adjacency_list)
        if adjacency_list: # if list is not empty
            unconnected_components.append(adjacency_list)
    return unconnected_components

def adjacency_matrix(X, K):
    N = X.shape[0]
    # Distance matrix first
    D = empty((N,N))
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i,j] = linalg.norm(X[i] - X[j])
            else:
                D[i,j] = inf
            
    # From distance matrix, find the K closest elements
    A = zeros((N,N))
    for i in range(N):
        # Gets the k indices that sorts ascending D[i]
        ids = argsort(D[i])
        k_ids = ids[:K]
        for idx in k_ids:
            A[i,idx] = 1.0
            A[idx,i] = 1.0
        A[i,i] = 0.0
    
    # Get unconnected components
    components = get_unconnected_components(A)
    
    # Check the closest edges between each component, being them different among them
    while(len(components) > 0):
        component_i = components.pop()
        min_dist = inf
        min_result = {"i" : -1, "j" : -1, "component_idx" : -1}
        for j in range(len(components)):
            result = components_distance(X, component_i, components[j])
            if(result["dist"] < min_dist):
                min_dist = result["dist"]
                min_result["i"] = result["c1_idx"]
                min_result["j"] = result["c2_idx"]
                min_result["component_idx"] = j
        if(min_result["component_idx"] == -1):
            continue
        # Connect
        A[min_result["i"], min_result["j"]] = 1.0
        A[min_result["j"], min_result["i"]] = 1.0
        new_component = component_i + components[min_result["component_idx"]]
        
        # Remove component_idx from the list of components
        del components[min_result["component_idx"]]
        
        # Add the new connection to the list of components
        components.append(new_component)
    
    return A


def degree(G):
    """
    Returns the degree of each vertex of an adjacency graph G. The degree matrix is a diagonal
    matrix in which each non-zero element contains the degree of the vertex
    """
    N = G.shape[0]
    D = zeros((N))
    for from_idx in range(N):
        degree = 0.0
        for for_idx in range(N):
            if(G[from_idx, for_idx] != 0):
                degree += 1.0
        D[from_idx] = degree
    return D

def laplacian_graph(X, K):
    A = adjacency_matrix(X, K)
    D = degree(A)
    N = A.shape[0]
    L = zeros((N,N))
    for i in range(N):
        for j in range(N):
            if (i != j) and A[i,j] != 0:
                L[i,j] = -1.0 / (1.0 * D[i])
            elif i == j:
                L[i,j] = 1.0
    return L

def conectivity_scatter(X, A):
    scatter(X[:,0], X[:,1])
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            if(A[i,j] != 0):
                arrow(X[i,0],X[i,1], X[j,0] - X[i,0], X[j,1] - X[i,1], shape='full', color='b')
    show()

"""
G = graph_utils.laplacian_graph(X)
clf()
fig, ax = subplots()
aa = ax.imshow(G)
cbar = fig.colorbar(aa)
"""