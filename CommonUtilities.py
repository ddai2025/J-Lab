"""
General utilities for moving data, performing statistical analysis, and more
in a file that can be imported into more specific notebooks.

Written by David D. Dai in collaboration with Minki Shim, spring 2025 MIT
8.13 "Junior Lab" staff, and Microsoft 365 Copilot.

J-Lab Python Introduction
https://github.mit.edu/juniorlab/Python-Intro/blob/master/pythonForJLAB.ipynb
"""


import pandas as pd
from numpy import *
from typing import Callable
from scipy import stats
from scipy import optimize as opt


# NEW
def txt_to_pd(path: str) -> pd.DataFrame:
    """
    Read a text file where data is separated by spaces into a DataFrame, 
    assuming that the top row specifies the column names.
    """

    # Read text file using Pandas (I pre-stripped junk from top of file).
    with open(path, "r") as file:
        lines = file.readlines()
    header = lines[0].strip().split()
    data_lines = [line.strip().split() for line in lines[1:]]
    return(pd.DataFrame(data_lines, columns=header))    


# NEW
def bin_events(events: ndarray, T: float) -> ndarray:
    """
    Given an array of timestamps for detector events, calculate the number
    of events that occured in each T-second interval.

    Params
    ----------
    events: float ndarray
        Timestamps for events.
    T: float
        Width of each time bin.

    Returns
    -------
    counts: float ndarray
        Number of events in each interval.
    """

    # Useful constants.
    max_t = max(events)
    num_intervals = int(ceil(max_t / T))
    event_counts = zeros(num_intervals, dtype=int)
    
    # Add each event to corresponding bin.
    for t in events:
        I = int(t // T)
        event_counts[I] += 1 
    return event_counts


# NEW
def integer_histogram(a: ndarray, width=-1) -> tuple[ndarray, ndarray]:
    """
    Special histogram function for positive integer data (e.g. the number
    of detector events in a fixed time interval). Special care is needed to
    avoid bins like [10.2, 10.7), which will NEVER have any data.

    Params
    ----------
    a: int ndarray
        Input data, something analogous to "number of detector events".
    width: int
        Width of each bin.

    Returns
    -------
    centers: int ndarray
        Centers of each bin.
    freqs: int ndarray
        Value of histogram.
    """

    assert a.dtype == int
    edges = arange(min(a) - 0.5, max(a) + 1.5, width)
    centers = (edges[:-1] + edges[1:]) / 2
    freqs, _ = histogram(a, bins=edges)
    return (centers, freqs)


# NEW
def running_average(x: ndarray) -> ndarray:
    """
    The (i)th running average is the average of x's first (i) elements, with
    (i) ranging from 1 to len(x) inclusive.
    """

    return cumsum(x) / arange(1, len(x) + 1)


# NEW
def calc_block_stats(x: ndarray, b: int) -> tuple[ndarray, ndarray]:
    """
    Split (x) into blocks of length (b) ignoring the tail of (x) and calculate
    the average and standard deviation of each block.

    Params
    ------
    x: ndarray
        The data to analyze.
    b: int
        Block length.

    Returns
    -------
    block_avgs: ndarray
        Average of each block.
    block_stds: ndarray
        Sample standard deviation, i.e.g we divide by (b - 1) instead of b.
    """

    N_blocks = len(x) // b
    x_blocks = x[:N_blocks * b].reshape(N_blocks, b)
    block_avgs = x_blocks.mean(axis=1)
    block_stds = x_blocks.std(axis=1, ddof=1)
    return block_avgs, block_stds


# NEW
def drop_zeros(centers: ndarray, freqs: ndarray) -> tuple[ndarray, ndarray]:
    """
    Drop histogram bins with zero counts for calculating chi-squared.

    Params
    ------
    centers: int ndarray
        Centers of each bin.
    freqs: int ndarray
        Value of histogram.
    

    Returns
    -------
    centers: int ndarray
        Centers of each bin excluding bins with zero counts.
    freqs: int ndarray
        Value of histogram excluding bins with zero counts.
    """

    keep = where(freqs != 0)
    return (centers[keep], freqs[keep])


# NEW
def calc_chi2(fit: Callable, fit_args: dict, x: ndarray, y: ndarray,
              y_stds: ndarray) -> float:
    """
    Params
    ------
    fit: Callable
        The fit / theoretical prediction for the data.
    fit_args: dict
        Additional arguments for the fit, e.g. Gaussian width and mean.
    x: ndarray
        Data.
    y: ndarray
        Data
    y_stds: ndarray
        Estimated errors for the data (y).
        
    Returns
    -------
    chi2: float
        Deviation of data from theory relative to the experimental error.
    """
    y_fit = fit(x, *fit_args)
    return sum((y - y_fit) ** 2 / y_stds ** 2)


# NEW
def calc_chi2_prob(chi2: float, ndof: int) -> float:
    """
    Calculate the chi-squared probability, i.e. the probability that a higher
    value of chi-squared would be obtained if the experiment were repeated.
    
    A small probability indicates that the fit is good, possibly "too good"
    because the estimated uncertanties were too small. A large probabilty
    indicates that the fit is poor, or that the estimated uncertanties are too
    large.

    Determining the number of degrees of freedom seems a little tricky. 
    According to the J-Lab Python Introduction, a Poisson fit has TWO degrees
    of freedom, even though the Poisson PMF only has one parameter "lambda". I
    guess that the normalization adds an additional freedom? Need to read more.

    Params
    ------
    chi2: float
        The value of chi-squared.
    ndof: int
        The number of degrees of freedom. This is a little tricky: according
        to the J-Lab Python Introduction, a Poisson fit has TWO degrees of
        freedom, while I personally expected one.

    Returns
    -------
    p_chi2: float
        The probability of getting a higher chi-squared.
    """

    p_chi2 = 1 - stats.chi2.cdf(chi2, ndof)
    return p_chi2


# NEW
def fit_chi2(fit: Callable, init_args: dict, x: ndarray, y: ndarray,
              y_stds: ndarray) -> tuple[ndarray, ndarray]:
    """
    Fit the data by minimizing the chi-squared. Uses SciPy optimization.

    Params
    ------
    fit: Callable
        The fit / theoretical prediction for the data.
    fit_args: dict
        Additional arguments for the fit, e.g. Gaussian width and mean.
    x: ndarray
        Data.
    y: ndarray
        Data
    y_stds: ndarray
        Estimated errors for the data (y).

    Returns
    -------
    opt_args: ndarray
        Fit arguments minimizing chi-squared.
    std_args: ndarray
        Uncertainty in the fit parameters.
    chi2: float
        Deviation of data from fit relative to the experimental error.
    p_chi2: float
        The probability of getting a higher chi-squared.
    """
    
    # Perform nonlinear least squares.
    opt_args, cov_args = opt.curve_fit(fit, x, y, sigma=y_stds, p0=init_args,
                                       absolute_sigma=True)
    std_args = sqrt(diag(cov_args))

    # Calculate chi-squared.
    chi2 = calc_chi2(fit, opt_args, x, y, y_stds)
    ndof = len(x) - 1 - len(opt_args) # Always -1 for normalization?
    p_chi2 = calc_chi2_prob(chi2, ndof)
    return (opt_args, std_args, chi2, p_chi2)

# NEW
def format_sigfigs(data: ndarray[float], stds: ndarray[float]) -> ndarray[str]:
    """
    Apply formating rules for significant figures, where we keep two sig figs
    of the error, e.g. 12.34 +/- 0.56.

    Params
    ------
    data: float ndarray
        Experimental data.
    stds: float ndarray
        Error (estimated standard deviation) of experimental data.

    Returns
    -------
    report: string ndarray
        Data and error written together in scientific notation with plus-minus
        notation and with correct significant figures.
    """

    report = []
    for d, s in zip(data, stds):
        l_d = int(floor(log10(abs(d))))
        l_s = int(floor(log10(abs(s))))
        d = round(d /  10 ** l_d, l_d - l_s + 1)
        s = round(s /  10 ** l_d, l_d - l_s + 1)
        d_fmt = eval(f"f'{{d:.{l_d - l_s + 1}f}}'")
        r = rf"$({d_fmt} ± {s})\times 10^{{{l_d}}}$"
        report.append(r)
    return array(report)