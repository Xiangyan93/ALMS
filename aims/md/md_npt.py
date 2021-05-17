#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
CWD = os.path.dirname(os.path.abspath(__file__))
DIR_DATA = os.path.join(CWD, '..', '..', 'data')
from ..args import MonitorArgs
from ..database import *
from ..aimstools.simulator.gromacs import Npt
from ..aimstools.jobmanager import Slurm


def get_NptSimulator(args: MonitorArgs) -> Npt:
    return Npt(packmol_exe=args.packmol_exe, dff_root=args.dff_root, gmx_exe_analysis=args.gmx_exe_analysis,
               gmx_exe_mdrun=args.gmx_exe_mdrun)


def create(args: MonitorArgs):
    for mol in session.query(Molecule).filter_by(active_learning=True).all():
        mol.create_md_npt(n_conformer=args.n_conformer)
    session.commit()


def build(args: MonitorArgs, simulator: Npt):
    for job in session.query(MD_NPT).filter_by(status=Status.STARTED).limit(args.n_prepare):
        simulator.build(path=job.ms_dir, smiles_list=[job.molecule.smiles], export=True)
        job.status = Status.PREPARED
        session.commit()
