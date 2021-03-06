"""ray.py
"""

import numpy as np
from obstacle import SimObject

class Ray:
    def __init__(self, p: np.array, r: float, intensity: float=0.0, ff_attenuation=1e-2):
        self.p = p
        self.r = r
        self.intensity = intensity
        self.ff_attenuation = ff_attenuation

    def cast(self, target: np.array, object_lst: [SimObject]):
        def checkIntersection(object: SimObject):
            """ Check if Ray intersect with object
            Thanks to Gareth Rees
            https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect/565282#565282
            """
            r = self.r
            s = object.s
            crs = np.cross(r, s)
            if crs == 0.:
                # Either ray is parallel or collinear with the object
                return False, np.inf
            u = np.cross(object.p - self.p, r) / crs
            if u >= 0. and u <= 1.:
                # Ray is in collision with object
                t = np.cross(object.p - self.p, s) / crs
                return True, t
            return False, np.inf
        
        while True:
            # Check for any collision on path and then extend the ray towards target
            target_t = (target - self.p) / self.r
            target_t = target_t[0]
            min_t = target_t
            min_obj = None
            for object in object_lst:
                has_collided, t = checkIntersection(object)
                if has_collided:
                    # Update minimum distance of collision
                    if t < min_t and t >= 0.:
                        min_t = t
                        min_obj = object
            min_t = min(min_t, target_t)
            # Compute intensity decay
            if min_t < 1e-4:
                intensity_decay = 1.
            else:
                intensity_decay = min(1., 1. / (min_t*self.ff_attenuation)**2)
            if min_obj is not None:
                # Compute intersection angle
                Ct_obj = np.dot(min_obj.s, self.r) / (np.linalg.norm(min_obj.s) * np.linalg.norm(self.r))
                # Get sine
                St_obj = np.sqrt(1 - Ct_obj**2)
                # Multiply object decay with intersection sine (thus normal rays will go further than parallel ones
                intensity_decay *= min_obj.get_decay() * St_obj
            # Apply intensity decay
            self.intensity *= intensity_decay
            # Move the ray
            self.p = self.p + min_t * self.r
            # Check if target is reached, else add 1e-3 towards self.r to avoid recasting to the same object
#            print(self.p, target, self.p - target, self.intensity, min_t)
            if (np.abs(self.p - target) < 1e-3).all() or min_t < 1e-5:
                return self.intensity
            self.p += 1e-3 * self.r
