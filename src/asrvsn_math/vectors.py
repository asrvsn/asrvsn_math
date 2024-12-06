'''
Functions
'''

import numpy as np
from typing import Tuple
import pdb

def vec_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    '''
    Angle between two vectors.
    '''
    v1_ = v1 / np.linalg.norm(v1)
    v2_ = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_, v2_), -1.0, 1.0))

def vec_angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    '''
    Angle between two vectors in degrees.
    '''
    return np.rad2deg(vec_angle(v1, v2))

def middle_mode(v: np.ndarray, bins: int=100) -> Tuple[float, float]:
    '''
    Middle mode of a distribution whose support vanishes from the ends.
    '''
    hist, edges = np.histogram(v, bins=bins)
    hist, edges = hist[1:-1], edges[1:-1]
    l = (hist != 0).argmax()
    r = hist.size - (hist[::-1] != 0).argmax()
    pdb.set_trace()
    return (edges[l], edges[r])