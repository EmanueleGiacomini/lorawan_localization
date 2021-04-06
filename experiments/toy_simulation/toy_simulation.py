"""toy_simulation.py
"""

import numpy as np
from ray import Ray
from obstacle import Wall2D

if __name__ == '__main__':
    for tpoint in target_points:
        # Generate a Ray from (0, 0) towards tpoint
        tpoint = np.float32(tpoint)
        ray_dir = tpoint.copy()
        ray_dir /= np.linalg.norm(ray_dir)
        ray = Ray(np.float32((0., 0.)), ray_dir, 10.)
        print(f'Perceived intensity at {tpoint}: {ray.cast(tpoint, []):.3f}')
    exit(0)
