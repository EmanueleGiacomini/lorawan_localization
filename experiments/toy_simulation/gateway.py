"""gateway.py
"""

import numpy as np
from ray import Ray
from obstacle import SimObject

class Gateway:
    def __init__(self, p: np.array, tx_power: float=100):
        self.p = p
        self.tx_power = tx_power

    def samplePoints(self, x, y, object_lst: [SimObject]=[]):
        output_intensity = np.zeros((x.shape[0], y.shape[0]))
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                output_intensity[i, j] = self.samplePoint(np.float32((x[i, j], y[i, j])), object_lst)
        return output_intensity

    def samplePoint(self, pt: np.array, object_lst: [SimObject]=[]):
        ray_dir = pt - self.p
        ray_dir /= np.linalg.norm(ray_dir)
        pt_intensity = Ray(self.p, ray_dir, self.tx_power).cast(pt, object_lst)
        #print(f'ray casted at {pt}: {pt_intensity}')
        return pt_intensity
