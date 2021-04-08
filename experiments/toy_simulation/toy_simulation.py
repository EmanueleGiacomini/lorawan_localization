"""toy_simulation.py
"""

import numpy as np
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
import matplotlib.patches as patches

from ray import Ray
from obstacle import Wall2D
from gateway import Gateway

def buildOffice0():
    object_lst = []
    # Outer strong walls
    object_lst.append(Wall2D(np.float32((-5,-5)), np.float32((-5,5)), decay=0.2))
    object_lst.append(Wall2D(np.float32((-5,5)), np.float32((5,5)), decay=0.2))
    object_lst.append(Wall2D(np.float32((5,5)), np.float32((5,-5)), decay=0.2))
    object_lst.append(Wall2D(np.float32((5,-5)), np.float32((-5,-5)), decay=0.2))
    # office no 1
    object_lst.append(Wall2D(np.float32((-5,0)), np.float32((-1,0)), decay=0.8))
    object_lst.append(Wall2D(np.float32((-1,0)), np.float32((-1,-2)), decay=0.8))
    object_lst.append(Wall2D(np.float32((-1,-4)), np.float32((-1,-5)), decay=0.8))
    # Office no 2
    object_lst.append(Wall2D(np.float32((2,-5)), np.float32((2,3)), decay=0.8))
    object_lst.append(Wall2D(np.float32((2,3)), np.float32((3,3)), decay=0.8))
    object_lst.append(Wall2D(np.float32((4,3)), np.float32((5,3)), decay=0.8))

    return object_lst

def drawSquare(p0: np.array, p1: np.array, decay=0.8, rect_idx_lst=[]):
    object_lst = []
    # Draw 4 walls
    p01 = np.float32((p0[0], p1[1]))
    p10 = np.float32((p1[0], p0[1]))
    object_lst.append(Wall2D(p0, p01, decay=decay))
    object_lst.append(Wall2D(p01, p1, decay=decay))
    object_lst.append(Wall2D(p1, p10, decay=decay))
    object_lst.append(Wall2D(p10, p0, decay=decay))
    if len(rect_idx_lst) == 0:
        rect_idx_lst.append(0)
    else:
        rect_idx_lst.append(rect_idx_lst[-1]+4)
    return object_lst
    
    
def buildCity0():
    object_lst = []
    rect_idx_lst = []
    object_lst.extend(drawSquare(np.float32((50, 0)), np.float32((150, 500)), decay=0.8, rect_idx_lst=rect_idx_lst))
    object_lst.extend(drawSquare(np.float32((310, 0)), np.float32((755, 83)), decay=0.8, rect_idx_lst=rect_idx_lst))
    object_lst.extend(drawSquare(np.float32((450, 250)), np.float32((900, 600)), decay=0.5, rect_idx_lst=rect_idx_lst))
    object_lst.extend(drawSquare(np.float32((50, -30)), np.float32((600, -150)), decay=0.7, rect_idx_lst=rect_idx_lst))
    object_lst.extend(drawSquare(np.float32((-25, -30)), np.float32((-800, -350)), decay=0.9, rect_idx_lst=rect_idx_lst))
    object_lst.extend(drawSquare(np.float32((-25, -30)), np.float32((-800, -350)), decay=0.9, rect_idx_lst=rect_idx_lst))
    object_lst.extend(drawSquare(np.float32((-30, 300)), np.float32((-800, 700)), decay=0.7, rect_idx_lst=rect_idx_lst))
    object_lst.extend(drawSquare(np.float32((50, -250)), np.float32((230, -600)), decay=0.5, rect_idx_lst=rect_idx_lst))
    object_lst.extend(drawSquare(np.float32((-30, -450)), np.float32((-350, -900)), decay=0.2, rect_idx_lst=rect_idx_lst))
    return object_lst, rect_idx_lst
    

if __name__ == '__main__':
    gate0 = Gateway(np.float32((300, 250)), tx_power=100)
    gate1 = Gateway(np.float32((-632, -613)), tx_power=150)
    gate2 = Gateway(np.float32((615, -732)), tx_power=150)

    #object_lst = buildOffice0()
    object_lst, rect_idx_lst = buildCity0()
    #object_lst = []

    # Create sampling space
    x = np.linspace(-1000, 1000, 40)
    y = np.linspace(-1000, 1000, 40)
    X, Y = np.meshgrid(x, y)

    def sampling_f(x, y):
        return gate0.samplePoints(x, y, object_lst)

    T = sampling_f(X, Y)
    T1 = gate1.samplePoints(X, Y, object_lst)
    T2 = gate2.samplePoints(X, Y, object_lst)

    npts = 400
    px, py = np.random.choice(x, npts), np.random.choice(y, npts)

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].contourf(X, Y, T,  levels=70 , cmap='hot', alpha=1)
    ax[0, 1].contourf(X, Y, T1,  levels=70 , cmap='hot', alpha=1)
    ax[1, 0].contourf(X, Y, T2,  levels=70 , cmap='hot', alpha=1)
    ax[1, 1].contourf(X, Y, T+T1+T2,  levels=70 , cmap='hot', alpha=1)
    #ax.contourf(X, Y, T1, levels=30, cmap='hot', alpha=0.33333)
    #ax.contourf(X, Y, T2, levels=30, cmap='hot', alpha=0.33333)
    
    for obj in object_lst:
        p0 = obj.p
        p1 = obj.p + obj.s
        ax[1, 1].plot([p0[0], p1[0]], [p0[1], p1[1]], 'k', lw=3)
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
        ax[1, 1].add_patch(patch)
        
    ax[0, 0].axis('equal')
    ax[0, 1].axis('equal')
    ax[1, 0].axis('equal')
    ax[1, 1].axis('equal')
    ax[0, 0].set_box_aspect(1)
    ax[0, 1].set_box_aspect(1)
    ax[1, 0].set_box_aspect(1)
    ax[1, 1].set_box_aspect(1)

    plt.show()
        
    exit(0)
