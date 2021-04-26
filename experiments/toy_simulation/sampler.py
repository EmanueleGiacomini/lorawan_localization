"""sampler.py
"""

import pickle
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from sklearn.preprocessing import normalize
import time

from ray import Ray
from obstacle import Wall2D
from gateway import Gateway
from toy_simulation import drawSquare

def build_cross():
    object_lst = pickle.load(open('./maps/cross00_object.p', 'rb'))
    # randomize intensity decay of objects
    for obj in object_lst:
        obj.decay = np.random.uniform()
    rect_idx_lst = pickle.load(open('./maps/cross00_rectidx.p', 'rb'))
    gate_lst = [Gateway(np.float32((10,10)), tx_power=100, ff_attenuation=1e-2),
                Gateway(np.float32((77,60)), tx_power=100, ff_attenuation=1e-2),
                Gateway(np.float32((74,7)), tx_power=100, ff_attenuation=1e-2)]
    return object_lst, rect_idx_lst, gate_lst

if __name__ == '__main__':
    np.random.seed(128)

    # Build obstacles
    object_lst, rect_idx_lst, gate_lst = build_cross()

    plt.style.use('dark_background')
    fig, ax = plt.subplots(1, figsize=(10, 8))


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
        patch = patches.Rectangle(p_min, w, h, linewidth=1, edgecolor='gray', facecolor='gray')
        ax.add_patch(patch)
    # Plot gateways
    for gateway in gate_lst:
        ax.scatter(gateway.p[0], gateway.p[1], marker='x', c='r')

    # Create sampling space
    #x = np.linspace(0, 100, 10)
    #y = np.linspace(0, 80, 10)
    #X, Y = np.meshgrid(x, y)
    #print(X.shape)
    #T_lst = [gate_lst[i].samplePoints(X, Y, object_lst) for i in range(len(gate_lst))]
    #ax.pcolormesh(X, Y, T_lst[0]+T_lst[1]+T_lst[2])

    ax.set_title('City cross')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.xlim([0, 100])
    plt.ylim([0, 80])

    sampled_pts = plt.ginput(200, timeout=0)
    sampled_pts = np.array([[p[0], p[1]] for p in sampled_pts])
    samp_x = sampled_pts[:, 0]
    samp_y = sampled_pts[:, 1]
    T0 = np.array([gate_lst[0].samplePoint(sampled_pts[i,:], object_lst) for i in range(samp_x.shape[0])])
    T1 = np.array([gate_lst[1].samplePoint(sampled_pts[i,:], object_lst) for i in range(samp_x.shape[0])])
    T2 = np.array([gate_lst[2].samplePoint(sampled_pts[i,:], object_lst) for i in range(samp_x.shape[0])])

    T = np.stack((T0, T1, T2), axis=1)
    T_color = T / 100.
    #T = normalize(T, axis=1)
    
    ax.scatter(samp_x, samp_y, c=T_color, edgecolor='none', s=np.max(T_color, axis=1)*150 )
    plt.pause(0.1)

    
    plt.tight_layout()
    #ax.pcolormesh(X, Y, T)
    #plt.pause(0.1)
    plt.show()

    sample_df = pd.DataFrame({'x': sampled_pts[:, 0], 'y': sampled_pts[:, 1], 't0': T[:,0], 't1': T[:, 1], 't2': T[:, 2]})
    out_filename = input('Type name of output file: ')
    sample_df.to_csv(out_filename)
    
    
    
    exit(0)
