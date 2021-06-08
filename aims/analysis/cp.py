#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import numpy as np
from numpy.polynomial.polynomial import polyval as np_polyval
from ..database.models import *
from aims.aimstools.utils import polyfit, polyval_derivative, is_monotonic
import matplotlib.pyplot as plt


def get_cp_inter(T_list: List[float], P_list: List[float], E_list: List[float],
                 algorithm: Literal['poly3'] = 'poly3') -> Optional[np.ndarray]:
    if len(set(P_list)) == 1:
        while True:
            if len(T_list) < 5:
                raise RuntimeError(f'{T_list}: data points less than 5.')
            if algorithm == 'poly3':
                coefs, score = polyfit(T_list, E_list, 4)
                if score > 0.999:
                    _, dEdT = polyval_derivative(T_list, coefs)
                    #plt.plot(T_list, E_list)
                    #plt.plot(T_list, np_polyval(T_list, coefs))
                    #plt.show()
                    return dEdT * 1000  # J/mol.K
                else:
                    print(score)
                    _, dEdT = polyval_derivative(T_list, coefs)
                    plt.plot(T_list, E_list)
                    plt.plot(T_list, np_polyval(T_list, coefs))
                    plt.show()
                    return None
    else:
        raise RuntimeError('TODO')


def get_cp_intra(T_list_in: List[float],
                 CV_list: List[float],
                 T_list_out: List[float]) -> np.ndarray:
    coefs, score = polyfit(T_list_in, CV_list, 4)
    return np_polyval(T_list_out, coefs)


def get_cp(mol: Molecule) -> Optional[Tuple[List[float], List[float], List[float]]]:
    jobs = [job for job in mol.md_npt if job.status == Status.ANALYZED]
    if len(jobs) < 5:
        return None
    n_mols = [json.loads(job.result)['n_mols'] for job in jobs]
    assert len(set(n_mols)) == 1
    einter = [json.loads(job.result)['einter'][0] / n_mols[0] for job in jobs]
    T_list = [job.T for job in jobs]
    P_list = [job.P for job in jobs]
    cp_inter = get_cp_inter(T_list, P_list, einter)
    if cp_inter is None:
        return None
    # intramolecular cp.
    jobs = [job for job in mol.qm_cv if job.status == Status.ANALYZED]
    if len(jobs) == 0:
        return None
    T_list_in = json.loads(jobs[0].result)['T']
    CV_list = json.loads(jobs[0].result)['cv_corrected']
    cp_intra = get_cp_intra(T_list_in=T_list_in, CV_list=CV_list, T_list_out=T_list)
    cp = (cp_inter + cp_intra).tolist()
    if not is_monotonic(cp):
        return None
    else:
        return T_list, P_list, cp


def update_fail_mols():
    for mol in session.query(Molecule).filter_by(active_learning=True):
        if Status.ANALYZED not in mol.status_qm_cv:
            continue
        status = mol.status_md_npt
        if len(status) > 2:
            continue
        elif not(len(status) == 2 and status[0] == Status.ANALYZED and status[1] == Status.FAILED):
            continue
        elif status[0] != Status.ANALYZED:
            continue

        if get_cp(mol) is None:
            mol.active_learning = False
            mol.fail = True
        session.commit()
