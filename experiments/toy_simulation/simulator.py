"""simulator.py
"""

import numpy as np

class Simulator:
    def __init__(self, map: (float, float)):
        """ Define a simulation environment.
        map: Dimension of the simulation map
        """
        self.map = np.float32(map)
