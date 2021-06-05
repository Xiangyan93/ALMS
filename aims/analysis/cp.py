#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import numpy as np
from numpy.polynomial.polynomial import polyval as np_polyval
from aims.aimstools.utils import polyfit, polyval_derivative


def get_cp_inter(T_list: List[float], P_list: List[float], E_list: List[float],
                 algorithm: Literal['poly3'] = 'poly3'):
    if len(set(P_list)) == 1:
        if len(T_list) < 5:
            raise RuntimeError(f'{T_list}: data points less than 5.')
        if algorithm == 'poly3':
            coefs, score = polyfit(T_list, E_list, 3)
            _, dEdT = polyval_derivative(T_list, coefs)
            return dEdT * 1000  # J/mol.K
    else:
        raise RuntimeError('TODO')


def get_cp_intra(T_list_in: List[float],
                 CV_list: List[float],
                 T_list_out: List[float]) -> np.ndarray:
    coefs, score = polyfit(T_list_in, CV_list, 4)
    return np_polyval(T_list_out, coefs)
