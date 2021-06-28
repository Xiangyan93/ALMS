#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import numpy as np
from numpy.polynomial.polynomial import polyval as np_polyval
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures


def polyfit(x: List[float], y: List[float], degree: int, weight: List[float] = None):
    """
    least squares polynomial fit
    when degree ==n, y = c0 + c1 * x + c2 * x**2 + ... + cn * x**n

    :param x:
    :param y:
    :param degree:
    :param weight:
    :return: coeff: array, np.array([c0, c1, ... , cn])
             score: int
    """
    skx = list(zip(x))
    skv = list(y)

    skx = np.array(skx).astype(np.float64)
    skv = np.array(skv).astype(np.float64)

    poly = PolynomialFeatures(degree)
    skx_ = poly.fit_transform(skx)
    clf = linear_model.LinearRegression(fit_intercept=False)
    clf.fit(skx_, skv, sample_weight=weight)
    return clf.coef_, clf.score(skx_, skv)


def polyval_derivative(x: Union[float, np.ndarray, List[float]],
                       coeff: List[float]) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    '''
    when degree == 2, len(coeff) = 3,
        y = c0 + c1 * x + c2 * xx
        dy/dx = c1 + 2*c2 * x
    when degree == 3, len(coeff) = 4,
        y = c0 + c1 * x + c2 * xx + c3 * xxx
        dy/dx = c1 + 2*c2 * x + 3*c3 * xx

    :param x:
    :param coeff: [c0, c1, c2, ...]
    :return: y: float
             dy/dx: float
    '''
    x = np.asarray(x)
    y = np_polyval(x, coeff)

    degree = len(coeff) - 1
    dydx = 0.
    for i in range(degree):
        dydx += (i + 1) * coeff[i + 1] * x ** i

    return y, dydx
