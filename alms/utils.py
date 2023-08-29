#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple


def get_T_list_from_range(t_min: float, t_max: float, n_point: int = 8) -> List[float]:
    t_min = int(t_min)
    t_max = int(t_max)
    t_span = t_max - t_min
    if t_max == t_min:
        return [t_min]
    if t_span <= 5:
        return [t_min, t_max]

    T_list = [t_min]
    while True:
        interval = t_span / (n_point - 1)
        if interval >= 5:
            break
        n_point -= 1

    for i in range(1, n_point):
        T_list.append(round(t_min + i * interval))

    return list(map(float, T_list))
