from matplotlib.pyplot import *
from matplotlib.collections import PatchCollection
from numpy import *

def draw_windows(
    window_clusters, 
    window_points, 
    window_dims, 
    window_cluster_ids, 
    n_clusters, 
    group_size_list,
    screen_height,
    screen_width,
    ticks_interval = 1
):
    clf()
    fig, ax = subplots(1, figsize=(10,5))
    ax.set_title("Clusters and Windows")
    ax.set_xlim(0, screen_width)
    ax.set_ylim(0, screen_height)
    ax.set_xticks(arange(0, screen_width, ticks_interval))
    ax.set_yticks(arange(0, screen_height, ticks_interval))
    
    # Draw Rectangles
    rect_clusters_list = [[] for i in range(n_clusters)]
    for j in range(sum(group_size_list)):
        point = window_points[j]
        dim = window_dims[j]
        idx = window_cluster_ids[j]
        rect = Rectangle((point[0], point[1]), dim[0], dim[1])
        rect_clusters_list[int(idx)].append(rect)
    color_list = ["cyan", "blue", "yellow", "green", "magenta", "gray", "black"]
    for i in range(n_clusters):
        rect_list = rect_clusters_list[i]
        pc = PatchCollection(rect_list, facecolor=color_list[i], edgecolor="k", alpha=0.2)
        ax.add_collection(pc)
        
    # Plot points
    colors_idx=window_cluster_ids[:].astype('int32')
    colors = [ color_list[color_idx] for color_idx in colors_idx]
    scatter(window_points[:,0], window_points[:,1], c=colors, alpha=0.3)

    # Plot Clusters
    colors_idx=arange(0,n_clusters,1)
    colors = [ color_list[color_idx] for color_idx in colors_idx]
    ax.scatter(window_clusters[:,0], window_clusters[:,1], c=colors, marker="*")
    return