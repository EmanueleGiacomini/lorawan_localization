"""obstacle.py
"""

import numpy as np
from abc import ABC, abstractmethod

class SimObject(ABC):
    def __init__(self):
        ...
    @abstractmethod
    def get_decay(self):
        ...

class Wall2D(SimObject):
    """ Wall ranges from p0 to p1.
    Class contains s term which represents the direction from p0 to p1
    Wall equation becomes:
        p + u * s 
    where u is in range [0., 1.]
    """
    def __init__(self, p0: np.array, p1: np.array, decay: float=0.2):
        self.p = p0
        self.s = p1 - p0
        self.decay = decay

    def get_decay(self):
        return self.decay
