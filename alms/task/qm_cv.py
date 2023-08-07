#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from abc import ABC

CWD = os.path.dirname(os.path.abspath(__file__))
DIR_DATA = os.path.join(CWD, '..', '..', 'data')
from tqdm import tqdm
from sqlalchemy.sql import or_
from simutools.simulator.gaussian.gaussian import GaussianSimulator
from .al import TaskAL
from ..args import MonitorArgs
from ..database.models import *
from ..aimstools.utils import create_dir
from ..analysis.cp import update_fail_mols


class TaskCV(TaskAL):
    def __init__(self, Gaussian: GaussianSimulator):
        self.simulator = Gaussian

    def create(self, args: MonitorArgs):
        create_dir(os.path.join(DIR_DATA, 'ms'))
        create_dir(os.path.join(DIR_DATA, 'slurm'))
        create_dir(os.path.join(DIR_DATA, 'tmp'))
        mols = session.query(SingleMoleculeTask).filter(
            or_(SingleMoleculeTask.active == True, SingleMoleculeTask.testset == True))
        for mol in tqdm(mols, total=mols.count()):
            fail_jobs = [job for job in mol.qm_cv if job.status == Status.FAILED]
            mol.create_qm_cv(n_conformer=args.n_conformer + len(fail_jobs))
        session.commit()

    def build(self, args: MonitorArgs):
        jobs = session.query(QM_CV).filter_by(status=Status.STARTED).limit(args.n_prepare)
        for job in tqdm(jobs, total=jobs.count()):
            job.commands = json.dumps(
                self.simulator.prepare(job.molecule.smiles, path=job.ms_dir, task='qm_cv',
                                       tmp_dir=os.path.join(DIR_DATA, 'tmp', str(job.id)), seed=job.seed)
            )
            job.status = Status.PREPARED
            session.commit()

    def run(self, args: MonitorArgs):
        n_submit = args.n_run - args.JobManager.n_current_jobs
        if n_submit > 0:
            for job in session.query(QM_CV).filter_by(status=Status.PREPARED).limit(n_submit):
                sh = args.JobManager.generate_sh(
                    name=job.name,
                    path=job.ms_dir,
                    commands=json.loads(job.commands),
                    partition=args.partition,
                    ntasks=args.n_cores,

                )
                args.JobManager.submit(sh)
                job.update_list('sh_file', [sh])
                job.status = Status.SUBMITED
                session.commit()

    def analyze(self, args: MonitorArgs):
        print('Analyzing results of qm_cv')
        args.JobManager.update_stored_jobs()
        jobs_to_analyze = session.query(QM_CV).filter_by(status=Status.SUBMITED).limit(args.n_analyze)
        for job in tqdm(jobs_to_analyze, total=jobs_to_analyze.count()):
            if not args.JobManager.is_running(job.slurm_name):
                result = self.simulator.analyze(os.path.join(job.ms_dir, 'gaussian.log'))
                if result is None or result == 'imaginary frequencies' or len(result['T']) == 0:
                    job.result = json.dumps(result)
                    job.status = Status.FAILED
                else:
                    job.result = json.dumps(result)
                    job.status = Status.ANALYZED
                session.commit()

    def extend(self, args: MonitorArgs):
        pass

    def update_fail_tasks(self):
        update_fail_mols()
