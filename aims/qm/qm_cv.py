#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
CWD = os.path.dirname(os.path.abspath(__file__))
DIR_DATA = os.path.join(CWD, '..', '..', 'data')
import json
from tqdm import tqdm
from ..args import MonitorArgs
from ..database import *
from ..aimstools.simulator.gaussian import GaussianSimulator
from ..aimstools.jobmanager import Slurm
from ..aimstools.utils import create_dir


def get_GaussianSimulator(args: MonitorArgs) -> GaussianSimulator:
    return GaussianSimulator(gauss_exe=args.gaussian_exe, n_jobs=args.n_cores, memMB=args.mem)


def create(args: MonitorArgs):
    # create dir.
    create_dir(os.path.join(DIR_DATA, 'ms'))
    create_dir(os.path.join(DIR_DATA, 'slurm'))
    create_dir(os.path.join(DIR_DATA, 'tmp'))
    # crete jobs.
    for mol in session.query(Molecule).filter_by(active_learning=True).all():
        fail_jobs = [job for job in mol.qm_cv if job.status == Status.FAILED]
        mol.create_qm_cv(n_conformer=args.n_conformer + len(fail_jobs))
    session.commit()


def build(args: MonitorArgs, simulator: GaussianSimulator):
    for job in session.query(QM_CV).filter_by(status=Status.STARTED).limit(args.n_prepare):
        job.commands = json.dumps(
            simulator.prepare(job.molecule.smiles, path=job.ms_dir, task='qm_cv',
                              tmp_dir=os.path.join(DIR_DATA, 'tmp', str(job.id)), seed=job.seed)
        )
        job.status = Status.PREPARED
        session.commit()


def run(args: MonitorArgs, simulator: GaussianSimulator, job_manager: Slurm):
    n_submit = args.n_run - job_manager.n_current_jobs
    if n_submit > 0:
        for job in session.query(QM_CV).filter_by(status=Status.PREPARED).limit(n_submit):
            sh = job_manager.generate_sh(path=job.ms_dir,
                                         name=job.name,
                                         commands=json.loads(job.commands))
            job_manager.submit(sh)
            job.update_list('sh_file', [sh])
            job.status = Status.SUBMITED
            session.commit()


def analyze(args: MonitorArgs, simulator: GaussianSimulator, job_manager: Slurm):
    print('Analyzing results of qm_cv')
    job_manager.update_stored_jobs()
    for job in tqdm(session.query(QM_CV).filter_by(status=Status.SUBMITED).limit(args.n_analyze), total=args.n_analyze):
        if not job_manager.is_running(job.slurm_name):
            result = simulator.analyze(os.path.join(job.ms_dir, 'gaussian.log'))
            if result is None or result == 'imaginary frequencies':
                job.result = result
                job.status = Status.FAILED
            else:
                job.result = json.dumps(result)
                job.status = Status.ANALYZED
            session.commit()


def extend(args: MonitorArgs, simulator: GaussianSimulator, job_manager: Slurm):
    pass
