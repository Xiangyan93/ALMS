#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
CWD = os.path.dirname(os.path.abspath(__file__))
DIR_DATA = os.path.join(CWD, '..', '..', 'data')
from ..args import MonitorArgs
from ..database import *
from ..aimstools.simulator.gaussian import GaussianSimulator
from ..aimstools.jobmanager import Slurm


def get_GaussianSimulator(args: MonitorArgs) -> GaussianSimulator:
    return GaussianSimulator(gauss_exe=args.gaussian_exe, n_jobs=args.n_cores, memMB=args.mem)


def create(args: MonitorArgs):
    for mol in session.query(Molecule).filter_by(active_learning=True).all():
        mol.create_qm_cv(n_conformer=args.n_conformer)
    session.commit()


def build(args: MonitorArgs, simulator: GaussianSimulator):
    for job in session.query(QM_CV).filter_by(status=Status.STARTED).limit(args.n_prepare):
        simulator.build(job.molecule.smiles, path=job.ms_dir, seed=job.seed)
        job.status = Status.PREPARED
        session.commit()


def run(args: MonitorArgs, simulator: GaussianSimulator, job_manager: Slurm):
    n_submit = args.n_run - job_manager.n_current_jobs
    if n_submit > 0:
        for job in session.query(QM_CV).filter_by(status=Status.PREPARED).limit(n_submit):
            cmds = simulator.get_slurm_commands(file=os.path.join(job.ms_dir, 'gaussian.gjf'),
                                                tmp_dir=os.path.join(DIR_DATA, 'tmp', str(job.id)))
            sh = job_manager.generate_sh(path=os.path.join(DIR_DATA, 'slurm'), name=job.name, commands=cmds)
            job_manager.submit(sh)
            job.update_list('sh_file', [sh])
            job.status = Status.SUBMITED
            session.commit()


def analyze(args: MonitorArgs, simulator: GaussianSimulator, job_manager: Slurm):
    for job in session.query(QM_CV).filter_by(status=Status.SUBMITED).limit(args.n_analyze):
        if not job_manager.is_running(job.name):
            result = simulator.analyze(os.path.join(job.ms_dir, 'gaussian.log'))
            if result is None:
                job.status = Status.FAILED
            else:
                job.update_dict('result', result)
                job.status = Status.ANALYZED
            session.commit()
