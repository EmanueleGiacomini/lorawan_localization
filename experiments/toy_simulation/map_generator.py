"""map_generator.py
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

    object_lst = []
    rect_idx_lst = []

    NUM_OBSTACLES = int(input('Type number of obstacles: '))
    
    for i in range(NUM_OBSTACLES):
        opts = plt.ginput(2)
        object_lst.extend(drawSquare(opts[0], opts[1], np.random.uniform(), rect_idx_lst))

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
        plt.pause(0.1)

    print(f'Placed {NUM_OBSTACLES} obstacles on the map.')
    fname = input('Insert name of the map: ')
    print(f'Saving {fname+"_object.p"} and {fname+"_rectidx.p"}')
    pickle.dump(object_lst, open('./maps/'+fname+'_object.p', 'wb'))
    pickle.dump(rect_idx_lst, open('./maps/'+fname+'_rectidx.p', 'wb'))
        
    exit(0)
