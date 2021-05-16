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
    return GaussianSimulator(gauss_exe=args.GAUSSIAN_EXE, n_jobs=args.n_cores, memMB=args.mem)


def get_JobManager(args: MonitorArgs) -> Slurm:
    return Slurm(partition=args.partition, n_nodes=args.n_nodes, n_cores=args.n_cores, n_gpu=args.n_gpu,
                 walltime=args.walltime)


def create(args: MonitorArgs):
    for mol in session.query(Molecule).filter_by(active_learning=True).all():
        mol.create_qm_cv(n_conformer=args.n_conformer)
    session.commit()


def prepare(args: MonitorArgs, simulator: GaussianSimulator):
    for job in session.query(QM_CV).filter_by(status=Status.STARTED).limit(args.n_prepare):
        simulator.prepare(job.molecule.smiles, path=job.ms_dir, seed=job.seed)
        job.status = Status.PREPARED
        session.commit()


def run(args: MonitorArgs, simulator: GaussianSimulator, job_manager: Slurm):
    n_submit = args.n_run - job_manager.n_current_jobs
    if n_submit > 0:
        for job in session.query(QM_CV).filter_by(status=Status.PREPARED).limit(n_submit):
            cmds = simulator.get_slurm_commands(file=os.path.join(job.ms_dir, 'gaussian.gjf'),
                                                tmp_dir=os.path.join(DIR_DATA, 'tmp', str(job.id)))
            job_manager.generate_sh(path=os.path.join(DIR_DATA, 'slurm'), name='qm_cv_%d' % job.id, commands=cmds)
