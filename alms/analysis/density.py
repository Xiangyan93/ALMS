#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from ..database.models import *
from alms.aimstools.utils import polyfit, is_monotonic, get_V


def get_density(mol: SingleMoleculeTask, plot_fail: bool = False) -> \
        Optional[Tuple[List[float], List[float], List[float]]]:
    jobs = [job for job in mol.md_npt_jobs if job.status == Status.ANALYZED]
    if len(jobs) < 5:
        return None
    n_mols = [json.loads(job.result)['n_mols'] for job in jobs]
    assert len(set(n_mols)) == 1
    T_list = [job.T for job in jobs]
    P_list = [job.P for job in jobs]
    # intermolecular cp.
    density_ = [json.loads(job.result)['density'][0] * 1000 for job in jobs]  # kg/m3
    density = get_V(T_list, P_list, density_, algorithm='poly2', r2_cutoff=0.98)
    if density is None:
        if plot_fail:
            import matplotlib.pyplot as plt
            plt.plot(T_list, density_)
            plt.show()
        return None
    density = density.tolist()
    if not is_monotonic(density):
        return None
    else:
        return T_list, P_list, density
