'''
Functions
'''

import numpy as np
from typing import Tuple
import pdb
import numpy.linalg as la

def vec_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    '''
    Angle between two vectors.
    '''
    return np.arccos(np.clip(cosine_similarity(v1, v2), -1.0, 1.0))

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
    return (edges[l], edges[r])

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.dot(v1, v2) / (la.norm(v1) * la.norm(v2))

def angle_between_lines(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.arccos(abs(np.clip(cosine_similarity(v1, v2), -1.0, 1.0)))
