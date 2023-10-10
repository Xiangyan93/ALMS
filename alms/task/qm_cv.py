#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tqdm import tqdm
from simutools.simulator.program import Gaussian
from .base import BaseTask
from ..args import MonitorArgs
from ..database.models import *
from ..analysis.cp import update_fail_mols


class TaskCV(BaseTask):
    def __init__(self, job_manager: Slurm, simulator: Gaussian):
        super().__init__(job_manager=job_manager)
        self.simulator = simulator

    def initiation(self, args: MonitorArgs):
        self.create_single_molecule_tasks()

    def active_learning(self, margs: MonitorArgs):
        super().active_learning(margs)

    def create(self, args: MonitorArgs):
        tasks = session.query(SingleMoleculeTask).filter(SingleMoleculeTask.active == True)
        for task in tqdm(tasks, total=tasks.count()):
            fail_jobs = [job for job in task.qm_cv if job.status == Status.FAILED]
            task.create_jobs(task='qm_cv', n_conformer=args.n_conformer + len(fail_jobs))
        session.commit()

    def build(self, args: MonitorArgs):
        jobs = session.query(QM_CV).filter_by(status=Status.STARTED).limit(args.n_prepare)
        for job in tqdm(jobs, total=jobs.count()):
            job.commands = json.dumps(
                self.simulator.prepare(job.single_molecule_task.molecule.smiles, path=job.ms_dir, task='qm_cv',
                                       tmp_dir=os.path.join(job.ms_dir, '../../../../../tmp', str(job.id)),
                                       seed=job.seed)
            )
            job.status = Status.PREPARED
            session.commit()

    def run(self, args: MonitorArgs):
        n_submit = args.n_run - self.job_manager.n_current_jobs
        if n_submit > 0:
            jobs_to_submit = session.query(QM_CV).filter_by(status=Status.PREPARED).limit(n_submit)
            self.submit_jobs(args=args, jobs_to_submit=jobs_to_submit)

    def analyze(self, args: MonitorArgs):
        jobs_to_analyze = self.get_jobs_to_analyze(QM_CV, n_analyze=args.n_analyze)
        results = self.analyze_multiprocess(self.analyze_single_job, jobs_to_analyze, args.n_jobs)
        for i, job in enumerate(jobs_to_analyze):
            result = results[i]
            job.result = json.dumps(result)
            if result is None or result == 'imaginary frequencies' or len(result['T']) == 0:
                job.status = Status.FAILED
            else:
                job.status = Status.ANALYZED
            session.commit()

    def analyze_single_job(self, job_dir: str):
        return self.simulator.analyze(os.path.join(job_dir, 'gaussian.log'))

    def extend(self, args: MonitorArgs):
        pass

    def update_fail_tasks(self):
        update_fail_mols()
