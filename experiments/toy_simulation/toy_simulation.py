"""toy_simulation.py
"""

import numpy as np
from scipy.interpolate import griddata
from matplotlib import pyplot as plt

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

def drawSquare(p0: np.array, p1: np.array, decay=0.8):
    object_lst = []
    # Draw 4 walls
    p01 = np.float32((p0[0], p1[1]))
    p10 = np.float32((p1[0], p0[1]))
    object_lst.append(Wall2D(p0, p01, decay=decay))
    object_lst.append(Wall2D(p01, p1, decay=decay))
    object_lst.append(Wall2D(p1, p10, decay=decay))
    object_lst.append(Wall2D(p10, p0, decay=decay))
    return object_lst
    
    
def buildCity0():
    object_lst = []
    object_lst.extend(drawSquare(np.float32((50, 0)), np.float32((150, 500)), decay=0.8))
    object_lst.extend(drawSquare(np.float32((310, 0)), np.float32((755, 83)), decay=0.8))
    object_lst.extend(drawSquare(np.float32((450, 250)), np.float32((900, 600)), decay=0.5))
    object_lst.extend(drawSquare(np.float32((50, -30)), np.float32((600, -150)), decay=0.7))
    object_lst.extend(drawSquare(np.float32((-25, -30)), np.float32((-800, -350)), decay=0.9))
    object_lst.extend(drawSquare(np.float32((-25, -30)), np.float32((-800, -350)), decay=0.9))
    object_lst.extend(drawSquare(np.float32((-30, 300)), np.float32((-800, 700)), decay=0.7))
    object_lst.extend(drawSquare(np.float32((50, -250)), np.float32((230, -600)), decay=0.5))
    object_lst.extend(drawSquare(np.float32((-30, -450)), np.float32((-350, -900)), decay=0.2))
    
    return object_lst
    

if __name__ == '__main__':
    gate0 = Gateway(np.float32((0, -15)), tx_power=100)

    #object_lst = buildOffice0()
    object_lst = buildCity0()
    #object_lst = []

    # Create sampling space
    x = np.linspace(-1000, 1000, 100)
    y = np.linspace(-1000, 1000, 100)
    X, Y = np.meshgrid(x, y)

    def sampling_f(x, y):
        return gate0.samplePoints(x, y, object_lst)

    T = sampling_f(X, Y)

    npts = 400
    px, py = np.random.choice(x, npts), np.random.choice(y, npts)

    fig, ax = plt.subplots(1, 1)
    ax.contourf(X, Y, T, levels=10)
    for obj in object_lst:
        p0 = obj.p
        p1 = obj.p + obj.s
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'k', lw=3)
    ax.axis('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.show()
        
    exit(0)
