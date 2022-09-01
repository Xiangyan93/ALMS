#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from ..database.models import *
from alms.aimstools.utils import polyfit, is_monotonic, get_V


def get_hvap(mol: Molecule, plot_fail: bool = False) -> Optional[Tuple[List[float], List[float], List[float]]]:
    jobs = [job for job in mol.md_npt
            if job.status == Status.ANALYZED and not np.isnan(json.loads(job.result)['einter'][0])]
    if len(jobs) < 5:
        return None
    n_mols = [json.loads(job.result)['n_mols'] for job in jobs]
    assert len(set(n_mols)) == 1
    T_list = [job.T for job in jobs]
    P_list = [job.P for job in jobs]
    # intermolecular cp.
    einter_ = [json.loads(job.result)['einter'][0] / n_mols[0] for job in jobs]
    einter = get_V(T_list, P_list, einter_, algorithm='poly2', r2_cutoff=0.98)
    if einter is None:
        if plot_fail:
            import matplotlib.pyplot as plt
            plt.plot(T_list, einter_)
            plt.show()
        return None
    else:
        hvap = (8.314 * np.asarray(T_list) / 1000 - einter).tolist()  # kJ/mol
    if not is_monotonic(hvap):
        return None
    else:
        return T_list, P_list, hvap
