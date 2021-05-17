#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import subprocess


def estimate_density_from_formula(f) -> float:
    # unit: g/mL
    from .formula import Formula
    formula = Formula.read(f)
    string = formula.to_str()
    density = {
        'H2': 0.07,
        'He': 0.15,
    }
    if string in density.keys():
        return density.get(string)

    nAtoms = formula.n_heavy + formula.n_h
    nC = formula.atomdict.get('C') or 0
    nH = formula.atomdict.get('H') or 0
    nO = formula.atomdict.get('O') or 0
    nN = formula.atomdict.get('N') or 0
    nS = formula.atomdict.get('S') or 0
    nF = formula.atomdict.get('F') or 0
    nCl = formula.atomdict.get('Cl') or 0
    nBr = formula.atomdict.get('Br') or 0
    nI = formula.atomdict.get('I') or 0
    nOther = nAtoms - nC - nH - nO - nN - nS - nF - nCl - nBr - nI
    return (1.175 * nC + 0.572 * nH + 1.774 * nO + 1.133 * nN + 2.184 * nS
            + 1.416 * nF + 2.199 * nCl + 5.558 * nBr + 7.460 * nI
            + 0.911 * nOther) / nAtoms


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


def random_string(length: int = 8):
    import random, string
    return ''.join(random.sample(string.ascii_letters, length))


def get_last_line(filename: str):
    # TODO implement windows version
    cmd = 'tail -n 1 %s' % filename
    try:
        out = subprocess.check_output(cmd.split()).decode()
    except:
        raise Exception('Cannot open file: %s' % filename)

    try:
        string = out.splitlines()[-1]
    except:
        string = ''
    return string
