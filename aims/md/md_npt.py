#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
CWD = os.path.dirname(os.path.abspath(__file__))
DIR_DATA = os.path.join(CWD, '..', '..', 'data')
import json
from ..args import MonitorArgs
from ..database import *
from ..aimstools.simulator.gromacs import Npt
from ..aimstools.jobmanager import Slurm


def get_NptSimulator(args: MonitorArgs) -> Npt:
    return Npt(packmol_exe=args.packmol_exe, dff_root=args.dff_root, gmx_exe_analysis=args.gmx_exe_analysis,
               gmx_exe_mdrun=args.gmx_exe_mdrun)


def create(args: MonitorArgs):
    for mol in session.query(Molecule).filter_by(active_learning=True).all():
        mol.create_md_npt()
    session.commit()


def build(args: MonitorArgs, simulator: Npt):
    for job in session.query(MD_NPT).filter_by(status=Status.STARTED).limit(args.n_prepare):
        path = os.path.join(job.molecule.ms_dir, 'md_npt', 'build')
        if not os.path.exists(path):
            os.mkdir(os.path.join(job.molecule.ms_dir, 'md_npt'))
            os.mkdir(path)

        if job.status == Status.STARTED:
            simulator.build(path=path, smiles_list=[job.molecule.smiles], export=True)
            for job_ in job.molecule.md_npt:
                job_.status = Status.BUILD
            session.commit()

        if job.status == Status.BUILD:
            job.commands = json.dumps(
                simulator.prepare(path=job.ms_dir, n_jobs=args.n_jobs, T=job.T, P=job.P, drde=True, T_basic=298)
            )
            job.status = Status.PREPARED
            session.commit()
