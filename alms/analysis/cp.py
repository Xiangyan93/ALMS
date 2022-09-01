#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import openbabel.pybel as pybel
from numpy.polynomial.polynomial import polyval as np_polyval
from ..database.models import *
from alms.aimstools.utils import polyfit, is_monotonic, get_V_dVdT


def get_cp_intra(T_list_in: List[float],
                 CV_list: List[float],
                 T_list_out: List[float]) -> np.ndarray:
    coefs, score = polyfit(T_list_in, CV_list, 4)
    return np_polyval(T_list_out, coefs)


def get_cp(mol: Molecule) -> Optional[Tuple[List[float], List[float], List[float], np.ndarray, np.ndarray, np.ndarray]]:
    jobs = [job for job in mol.md_npt
            if job.status == Status.ANALYZED and not np.isnan(json.loads(job.result)['einter'][0])]
    if len(jobs) < 5:
        return None
    n_mols = [json.loads(job.result)['n_mols'] for job in jobs]
    assert len(set(n_mols)) == 1
    # pv cp
    T_list = [job.T for job in jobs]
    P_list = [job.P for job in jobs]
    density = [json.loads(job.result)['density'][0] for job in jobs]
    molwt = pybel.readstring('smi', mol.smiles).molwt
    _ = get_V_dVdT(T_list, P_list, density, algorithm='poly2', r2_cutoff=0.98)
    if _ is None:
        return None
    else:
        cp_pv = - molwt * np.asarray(P_list) * _[1] * 0.1 / np.asarray(density) ** 2  # J/mol.K
    # intermolecular cp.
    einter = [json.loads(job.result)['einter'][0] / n_mols[0] for job in jobs]
    _ = get_V_dVdT(T_list, P_list, einter, algorithm='poly2', r2_cutoff=0.98)
    if _ is None:
        return None
    else:
        cp_inter = _[1] * 1000  # J/mol.K
    # intramolecular cp.
    jobs = [job for job in mol.qm_cv if job.status == Status.ANALYZED]
    if len(jobs) == 0:
        return None
    T_list_in = json.loads(jobs[0].result)['T']
    CV_list = json.loads(jobs[0].result)['cv_corrected']
    cp_intra = get_cp_intra(T_list_in=T_list_in, CV_list=CV_list, T_list_out=T_list)
    cp = (cp_inter + cp_intra + cp_pv).tolist()
    if not is_monotonic(cp):
        return None
    else:
        return T_list, P_list, cp, cp_inter, cp_intra, cp_pv


def update_fail_mols():
    for mol in session.query(Molecule).filter_by(active=True):
        # at least one QM job is success.
        if Status.ANALYZED not in mol.status_qm_cv:
            continue
        status = mol.status_md_npt
        # MD jobs must be all ANALYZED or FAILED.
        if len(status) > 2 or len(status) == 0:
            continue
        elif len(status) == 2:
            if not(status[0] == Status.ANALYZED and status[1] == Status.FAILED):
                continue
        elif status[0] != Status.ANALYZED:
            continue

        if get_cp(mol) is None:
            mol.active = False
            mol.inactive = True
            mol.fail = True
        session.commit()
