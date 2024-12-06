'''
Statistical functions
'''
from typing import Callable, Tuple
import numpy as np
from tqdm import tqdm

def mean_stderr(data: np.ndarray) -> Tuple[float, float]:
    '''
    Compute mean and standard error of the mean of data.
    '''
    assert data.ndim == 1, 'Data must be 1D'
    return data.mean(), data.std() / np.sqrt(data.shape[0])

def reject_outliers(arr: np.ndarray, start_perc: float, end_perc: float) -> np.ndarray:
    '''
    Reject outliers from array by percentile
    '''
    start_val = np.percentile(arr, start_perc)
    end_val = np.percentile(arr, end_perc)
    return arr[(arr >= start_val) & (arr <= end_val)]

def mad_reject_xy(xdata: np.ndarray, ydata: np.ndarray, mad_thresh: float=3.0) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Reject outliers from xdata and ydata by median absolute deviation
    '''
    assert xdata.shape == ydata.shape, 'xdata and ydata must have same shape'
    x_keep_mask = mad_outlier_mask(xdata, thresh=mad_thresh)
    y_keep_mask = mad_outlier_mask(ydata, thresh=mad_thresh)
    keep_mask = np.logical_and(x_keep_mask, y_keep_mask)
    xdata = xdata[keep_mask]
    ydata = ydata[keep_mask]
    assert xdata.shape == ydata.shape, 'xdata and ydata must have same shape after outlier rejection'
    return xdata, ydata

def mad_reject(data: np.ndarray, mad_thresh: float=3.5) -> np.ndarray:
    '''
    Reject outliers from data by median absolute deviation
    '''
    return data[mad_outlier_mask(data, thresh=mad_thresh)]

def mad_outlier_mask(points: np.ndarray, thresh=3.5) -> np.ndarray:
    """
    See https://stackoverflow.com/questions/11882393/matplotlib-disregard-outliers-when-plotting

    Returns a boolean array with True if points are not outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation # using MAD-estimated stdev

    return modified_z_score <= thresh

def parametric_bootstrap(
        estimator: Callable[[np.ndarray], float],
        sampler: Callable[[], np.ndarray],
        percentiles: Tuple[float, float]=(2.5, 97.5),
        n_samples: int=1000,
    ) -> Tuple[float, float]:
    '''
    Perform parametric bootstrap to estimate confidence intervals of estimator.
    '''
    estimates = []
    for _ in tqdm(range(n_samples)):
        # Generate bootstrap sample
        sample = sampler()
        estimate = estimator(sample)
        estimates.append(estimate)
    estimates = np.array(estimates)
    return np.percentile(estimates, percentiles)