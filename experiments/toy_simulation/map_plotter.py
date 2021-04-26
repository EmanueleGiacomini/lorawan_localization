"""map_plotter.py
"""

import pickle
import numpy as np
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import time

from ray import Ray
from obstacle import Wall2D
from gateway import Gateway
from toy_simulation import drawSquare

if __name__ == '__main__':
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.set_title('City cross')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.xlim([0, 100])
    plt.ylim([0, 80])
    plt.tight_layout()
    
    fname = input('Type map name: ')
    object_lst = pickle.load(open('./maps/'+fname+'_object.p', 'rb'))
    rect_idx_lst = pickle.load(open('./maps/'+fname+'_rectidx.p', 'rb'))

    # Add patches on buildings
    for bidx in rect_idx_lst:
        # get first and last point
        rect_lst = object_lst[bidx:bidx+4]
        p_mat = np.concatenate([o.p for o in rect_lst], axis=0)
        p_mat = np.reshape(p_mat, (-1, 2))
        p_min = np.float32((np.min(p_mat[:,0]), np.min(p_mat[:,1])))
        p_max = np.float32((np.max(p_mat[:,0]), np.max(p_mat[:,1])))
        h = p_max[1] - p_min[1]
        w = p_max[0] - p_min[0]
        patch = patches.Rectangle(p_min, w, h, linewidth=1, edgecolor='k', facecolor='k')
        ax.add_patch(patch)

    plt.show()
        
    exit(0)
