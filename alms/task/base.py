#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Callable
import math
from tqdm import tqdm
from multiprocessing import Pool
from alms.database.models import *
from .abc import ABCTask
from ..args import MonitorArgs


class BaseTask(ABCTask):
    def __init__(self, job_manager: Slurm):
        self.job_manager = job_manager

    def active_learning(self, margs: MonitorArgs):
        if margs.strategy == 'all':
            if margs.task in ['qm_cv', 'md_npt']:
                mols_all = session.query(SingleMoleculeTask)
            elif margs.task == 'md_binding':
                mols_all = session.query(DoubleMoleculeTask)
            else:
                raise ValueError
            for mol in tqdm(mols_all.all(), total=mols_all.count()):
                mol.active = True
                mol.inactive = False
            session.commit()
        """
        if margs.stop_uncertainty is None:
            return
        if margs.stop_uncertainty < 0.0:
            mols_all = session.query(SingleMolecule)
            for mol in tqdm(mols_all.all(), total=mols_all.count()):
                mol.active = True
                mol.inactive = False
            session.commit()
            return
        args = ActiveLearningArgs()
        args.dataset_type = 'regression'
        args.save_dir = '../data'
        n_jobs = mn_jobs
        args.pure_columns = ['smiles']
        args.target_columns = ['target']
        args.graph_kernel_type = margs.graph_kernel_type
        args.graph_hyperparameters = ['data/tMGR.json']
        args.model_type = 'gpr'
        args.alpha = 0.01
        args.optimizer = None
        args.batch_size = None
        args.learning_algorithm = 'unsupervised'
        args.add_size = 1
        args.cluster_size = 1
        args.pool_size = margs.pool_size
        args.stop_size = 100000
        args.stop_uncertainty = [margs.stop_uncertainty]
        args.evaluate_stride = 0
        args.seed = margs.seed

        mols_all = session.query(SingleMolecule).filter_by(fail=False)
        # random select 2 samples as the start of active learning.
        if mols_all.filter_by(active=True).count() <= 1:
            for mol in np.random.choice(mols_all.all(), 2, replace=False):
                mol.active = True
                mol.inactive = False
            session.commit()

        # get selected data set.
        mols = mols_all.filter_by(active=True)
        df = pd.DataFrame({'smiles': [mol.smiles for mol in mols],
                           'target': [0.] * mols.count()})
        dataset = Dataset.from_df(args, df)
        dataset.update_args(args)
        # get pool data set.
        mols = mols_all.filter_by(active=False, inactive=False).limit(50000)
        if mols.count() == 0:
            return
        df_pool = pd.DataFrame({'smiles': [mol.smiles for mol in mols],
                                'target': [0.] * mols.count()})
        dataset_pool = Dataset.from_df(args, df_pool)
        dataset_pool.update_args(args)
        # get full data set.
        dataset_full = dataset.copy()
        dataset_full.data = dataset.data + dataset_pool.data
        dataset_full.unify_datatype(dataset_full.X_graph)
        #
        kernel_config = get_kernel_config(args, dataset_full, kernel_pkl=os.path.join(args.save_dir, 'kernel.pkl'))
        # active learning
        al = ActiveLearner(args, dataset, dataset_pool, kernel_config, kernel_config)
        al.run()
        if len(al.dataset) != 0:
            smiles = [s.split(',')[0] for s in al.dataset.X_repr.ravel()]
            for mol in tqdm(mols_all.all(), total=mols_all.count()):
                if mol.smiles in smiles:
                    mol.active = True
                    mol.inactive = False
        if len(al.dataset_pool) != 0:
            smiles = [s.split(',')[0] for s in al.dataset_pool.X_repr.ravel()]
            for mol in tqdm(mols_all.all(), total=mols_all.count()):
                if mol.smiles in smiles:
                    mol.active = False
                    mol.inactive = True
        session.commit()
        """

    @staticmethod
    def create_single_molecule_tasks():
        mols = session.query(Molecule)
        for mol in tqdm(mols, total=mols.count()):
            task = SingleMoleculeTask(molecule_id=mol.id)
            add_or_query(task, ['molecule_id'])
        session.commit()

    @staticmethod
    def create_double_molecule_tasks():
        mols = session.query(Molecule)
        for i, mol1 in enumerate(mols):
            for j in range(i, mols.count()):
                mol2 = mols[j]
                mid = f'{mol1.id}_{mol2.id}'
                task = DoubleMoleculeTask(molecules_id=mid)
                add_or_query(task, ['molecules_id'])
        session.commit()

    def submit_jobs(self, jobs_to_submit: List, extend: bool = False):
        if jobs_to_submit.count() != 0:
            for job in jobs_to_submit:
                name = job.name + '_extend' if extend else job.name
                commands = json.loads(job.commands_extend) if extend else json.loads(job.commands_mdrun)
                sh = self.job_manager.generate_sh(path=job.ms_dir,
                                                  name=name,
                                                  commands=commands,
                                                  sh_index=True)
                self.job_manager.submit(sh)
                job.update_list('sh_file', [sh])
                job.status = Status.SUBMITED
                session.commit()

    def get_jobs_to_analyze(self, table, n_analyze: int) -> List:
        self.job_manager.update_stored_jobs()
        jobs_to_analyze = []
        for job in session.query(table).filter_by(status=Status.SUBMITED).limit(n_analyze):
            if not self.job_manager.is_running(job.slurm_name):
                jobs_to_analyze.append(job)
        return jobs_to_analyze

    @staticmethod
    def analyze_multiprocess(analyze_function: Callable, jobs_to_analyze: List, n_jobs: int = 1) -> List:
        results = []
        n_analyze = int(math.ceil(len(jobs_to_analyze) / n_jobs))
        for i in tqdm(range(n_analyze), total=n_analyze):
            jobs = jobs_to_analyze[i * n_jobs:(i + 1) * n_jobs]
            with Pool(n_jobs) as p:
                result = p.map(analyze_function, jobs)
            results += result
        return results
