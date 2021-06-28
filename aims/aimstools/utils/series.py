#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import math
import numpy as np
from pandas import Series
from .fitting import polyfit, np_polyval, polyval_derivative


def block_average(series: Series, n_block=5) -> (float, float):
    '''
    Get block average and standard error
    '''
    block_aves = average_of_blocks(series, n_block)
    ave, stderr = np.mean(block_aves), np.std(block_aves, ddof=1) / math.sqrt(n_block)
    stderr = float('%.1e' % stderr)  # 2 effective number for stderr
    return ave, stderr


def average_of_blocks(series: Series, n_block=5) -> [float]:
    '''
    Split data to several blocks and return the average of each block
    '''
    n_points = len(series)
    block_size = n_points // n_block
    blocks = []
    for n in range(n_block - 1):
        blocks.append(series.iloc[block_size * n:block_size * (n + 1)])
    blocks.append(series.iloc[block_size * (n_block - 1):])
    block_aves = [np.mean(b) for b in blocks]
    return block_aves


def is_converged(series: Series, frac_min=0.5) -> (bool, float):
    from pymbar import timeseries

    n_points = len(series)
    array = np.array(series)
    t0, g, Neff_max = timeseries.detectEquilibration(array, nskip=max(1, n_points // 100))
    if t0 > n_points * (1 - frac_min):
        return False, series.index[t0]
    return True, series.index[t0]


def efficiency_with_block_size(l: List[float]) -> [float]:
    array = np.array(l)
    n_points = len(l)
    bsize_list = []
    n_block_list = []
    s_list = []

    for bsize in range(1, int(math.sqrt(n_points))):
        n_block = int(n_points / bsize)
        bsize_list.append(bsize)
        n_block_list.append(n_block)

    for n_block in range(int(math.sqrt(n_points)), 4, -1):
        bsize = int(n_points / n_block)
        bsize_list.append(bsize)
        n_block_list.append(n_block)

    for i, bsize in enumerate(bsize_list):
        n_block = n_block_list[i]
        blocks = np.array_split(array, n_block)
        ave_blocks = [np.mean(block) for block in blocks]
        std_ave_blocks = np.std(ave_blocks, ddof=1)
        s = bsize * std_ave_blocks ** 2 / np.std(array) ** 2
        s_list.append(s)

    import pylab
    pylab.plot(bsize_list, s_list, '.')
    pylab.show()


def mean_and_uncertainty(series: Series, inefficiency=None) -> (float, float):
    from pymbar import timeseries

    ave = np.mean(series)
    array = np.array(series)
    if inefficiency == None:
        inefficiency = timeseries.statisticalInefficiency(array)
    return ave, np.std(array, ddof=1) / math.sqrt(len(array) / inefficiency)


def is_monotonic(values: List[float]) -> bool:
    m = (values[1] - values[0] > 0)
    for i in range(2, len(values)):
        if (values[i] - values[i-1] > 0) ^ m:
            return False
    return True


def get_longest_monotonic_list(values: List[float], positive: bool = True,
                               return_start_end: bool = False) -> List[float]:
    """get longest monotonic list."""
    start = 0
    length = 1
    if positive:
        for i in range(len(values)):
            print(i)
            if len(values) - i <= length:
                break

            for j in range(i+1, len(values)):
                if values[j] - values[j-1] > 0:
                    continue
                else:
                    if j - i > length:
                        start = i
                        length = j - i
                    break
            else:
                start = i
                length = len(values) - i
    else:
        result = get_longest_monotonic_list([-v for v in values], positive=True)
        return [-v for v in result]
    if return_start_end:
        return [start, start+length]
    else:
        return values[start:start+length]


def get_V(T_list: List[float],
          P_list: List[float],
          V_list: List[float],
          algorithm: Literal['poly2'] = 'poly2',
          r2_cutoff: float = 0.98) -> Optional[np.ndarray]:
    if len(set(P_list)) == 1:
        while True:
            if len(T_list) < 5:
                raise RuntimeError(f'{T_list}: data points less than 5.')
            if algorithm.startswith('poly'):
                coefs, score = polyfit(T_list, V_list, int(algorithm[4:]))
                if score > r2_cutoff:
                    return np_polyval(T_list, coefs)
                else:
                    return None
    else:
        raise RuntimeError('TODO')


def get_V_dVdT(T_list: List[float],
               P_list: List[float],
               V_list: List[float],
               algorithm: Literal['poly2'] = 'poly2',
               r2_cutoff: float = 0.98) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if len(set(P_list)) == 1:
        while True:
            if len(T_list) < 5:
                raise RuntimeError(f'{T_list}: data points less than 5.')
            if algorithm.startswith('poly'):
                coefs, score = polyfit(T_list, V_list, int(algorithm[4:]))
                if score > r2_cutoff:
                    V, dVdT = polyval_derivative(T_list, coefs)
                    return V, dVdT
                else:
                    return None
    else:
        raise RuntimeError('TODO')
