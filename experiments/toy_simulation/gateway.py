"""gateway.py
"""

import numpy as np
import multiprocessing
import time
from multiprocessing.pool import ThreadPool
from ray import Ray
from obstacle import SimObject

class Gateway:
    def __init__(self, p: np.array, tx_power: float=100, ff_attenuation=1e-2):
        self.p = p
        self.tx_power = tx_power
        self.ff_attenuation=ff_attenuation

    def samplePoints(self, x, y, object_lst: [SimObject]=[], parallelize=False):
        output_intensity = np.zeros((x.shape[0], y.shape[0]))
        if parallelize is False:
            for i in range(x.shape[0]):
                for j in range(y.shape[0]):
                    output_intensity[i, j] = self.samplePoint(np.float32((x[i, j], y[i, j])), object_lst)
            return output_intensity
        else:
            # Parallelized version
            num_cores = multiprocessing.cpu_count()
            x_vect = np.reshape(x, -1)
            y_vect = np.reshape(y, -1)
            output_vect = np.reshape(output_intensity, -1)
            # Split the x and y arrays into num_cores chunks
            x_split_lst = np.array_split(x_vect, num_cores)
            y_split_lst = np.array_split(y_vect, num_cores)
            output_split_lst = np.array_split(output_vect, num_cores)
            # Define thread function to handle a chunk of data
            def thread_fn(j):
                start = time.time()
                print(f'Thread {j} handling {x_split_lst[j].shape[0]} elements')
                # Run samplePoint function on the j-th chunk
                x_vect_j = x_split_lst[j]
                y_vect_j = y_split_lst[j]
                output_vect_j = output_split_lst[j]
                for i in range(x_split_lst[j].shape[0]):
                    output_vect_j[i] = self.samplePoint(np.float32((x_vect_j[i], y_vect_j[i])), object_lst)
                end = time.time()
                print(f'Thread {j} took {end-start} s')
                return
            # Define pool object
            pool = ThreadPool(num_cores)
            pool.map(thread_fn, range(len(x_split_lst)))

            # Once output_split_lst contains all the data, reassemble the vector and reshape into (x.shape[0], y.shape[0])
            output_vect = np.concatenate(output_split_lst)
            output_vect = np.reshape(output_vect, (x.shape[0], y.shape[0]))
            return output_vect

    def samplePoint(self, pt: np.array, object_lst: [SimObject]=[]):
        ray_dir = pt - self.p
        ray_dir /= np.linalg.norm(ray_dir)
        pt_intensity = Ray(self.p, ray_dir, self.tx_power, self.ff_attenuation).cast(pt, object_lst)
        #print(f'ray casted at {pt}: {pt_intensity}')
        return pt_intensity
