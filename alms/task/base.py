#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os.path
from abc import ABC
from typing import Callable
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from molalkit.args import ActiveLearningArgs
from molalkit.al.learner import ActiveLearner
from alms.database.models import *
from alms.task.abc import ABCTask
from alms.args import MonitorArgs


class BaseTask(ABCTask, ABC):
    def __init__(self, job_manager: Slurm):
        self.job_manager = job_manager

    def _init_active_tasks(self, margs: MonitorArgs, tasks_all):
        tasks_active = tasks_all.filter_by(active=True, inactive=False)
        if tasks_active.count() == 0:
            if tasks_all.count() == 0:
                raise ValueError(f'Task {margs.task} not found. Please submit molecules first.')
            elif tasks_all.count() == 1:
                tasks_all.first().active = True
                tasks_all.first().inactive = False
            else:
                # randomly sample 2 data as the start of active learning.
                tasks_active = np.random.choice(tasks_all.all(), 2, replace=False)
                for task in tasks_active:
                    task.active = True
                    task.inactive = False
                    task.selected_id = 0
                tasks_active = tasks_all.filter_by(active=True, inactive=False)
        return tasks_active

    def active_learning(self, margs: MonitorArgs):
        save_dir = f'{CWD}/../../data/simulation/al_tmp'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Active learning loops
        # In each iteration, n_query data are queried from the pool set to save the memory.
        while True:
            simulation_finished = True
            if margs.task in ['qm_cv', 'md_npt', 'md_solvation']:
                pure_columns = ['smiles']
                tasks_all = session.query(SingleMoleculeTask)
                tasks_active = self._init_active_tasks(margs, tasks_all)
                df_active = pd.DataFrame({'smiles': [], 'target': [], 'id': []})
                for task in tasks_active:
                    if task.properties is None or json.loads(task.properties).get(margs.property_name) is None:
                        simulation_finished = False
                        target = 0.
                    else:
                        target = json.loads(task.properties).get(margs.property_name)
                    df_active.loc[len(df_active)] = task.molecule.smiles, target, task.id
                df_active.to_csv(f'{save_dir}/train.csv', index=False)
                tasks_pool = tasks_all.filter_by(active=False, inactive=False)
                if margs.n_query is not None:
                    tasks_pool = tasks_pool.limit(margs.n_query)
                if tasks_pool.count() == 0:
                    return
                df_pool = pd.DataFrame({'smiles': [], 'target': [], 'id': []})
                for task in tasks_pool:
                    # set the target property to be 0. before the simulation is finished.
                    df_pool.loc[len(df_pool)] = task.molecule.smiles, 0., task.id
                df_pool.to_csv(f'{save_dir}/pool.csv', index=False)
            elif margs.task == 'md_binding':
                pure_columns = ['smiles1', 'smiles2']
                tasks_all = session.query(DoubleMoleculeTask)
                tasks_active = self._init_active_tasks(margs, tasks_all)
                df_active = pd.DataFrame({'smiles1': [], 'smiles2': [], 'target': [], 'id': []})
                for task in tasks_active:
                    if task.properties is None or json.loads(task.properties).get(margs.property_name) is None:
                        simulation_finished = False
                        target = 0.
                    else:
                        target = json.loads(task.properties).get(margs.property_name)
                    smiles1, smiles2 = task.molecule_1.smiles, task.molecule_2.smiles
                    df_active.loc[len(df_active)] = smiles1, smiles2, target, task.id
                df_active.to_csv(f'{save_dir}/train.csv', index=False)
                tasks_pool = tasks_all.filter_by(active=False, inactive=False)
                if margs.n_query is not None:
                    tasks_pool = tasks_pool.limit(margs.n_query)
                if tasks_pool.count() == 0:
                    return
                df_pool = pd.DataFrame({'smiles1': [], 'smiles2': [], 'target': [], 'id': []})
                for task in tasks_pool:
                    smiles1, smiles2 = task.molecule_1.smiles, task.molecule_2.smiles
                    df_pool.loc[len(df_pool)] = smiles1, smiles2, 0., task.id
                df_pool.to_csv(f'{save_dir}/pool.csv', index=False)
            else:
                raise ValueError(f'unknown task: {margs.task}')
            # skip active learning if the simulation is not finished.
            simulation_finished = True
            if tasks_active.count() != 2 and not simulation_finished:
                return

            if margs.learning_type == 'all':
                tasks_unselected = tasks_all.filter_by(active=False, inactive=False)
                for task in tqdm(tasks_unselected.all(), total=tasks_unselected.count()):
                    task.active = True
                    task.selected_id = 0
                session.commit()
                return
            elif margs.learning_type == 'explorative_gpr_pu':
                arguments = [
                    '--data_path_training', f'{save_dir}/train.csv',
                    '--data_path_pool', f'{save_dir}/pool.csv',
                    '--dataset_type', 'regression',
                    '--pure_columns', 'smiles1', 'smiles2',
                    '--target_columns', 'target',
                    '--learning_type', 'explorative',
                    '--batch_size', str(margs.batch_size),
                    '--stop_cutoff', str(margs.stop_cutoff),
                    '--model_config_selector', margs.model_config,
                    '--save_dir', save_dir,
                    '--n_jobs', str(margs.n_jobs),
                    '--pure_columns'
                ] + pure_columns
                if margs.n_query is not None:
                    arguments += ['--n_query', str(margs.n_query)]
                args = ActiveLearningArgs().parse_args(arguments)
                active_learner = ActiveLearner(save_dir=args.save_dir,
                                            selection_method=args.selection_method,
                                            forgetter=args.forgetter,
                                            model_selector=args.model_selector,
                                            dataset_train_selector=args.data_train_selector,
                                            dataset_pool_selector=args.data_pool_selector,
                                            dataset_val_selector=args.data_val_selector,
                                            metrics=args.metrics,
                                            top_k_id=args.top_k_id,
                                            model_evaluators=args.model_evaluators,
                                            dataset_train_evaluators=args.data_train_evaluators,
                                            dataset_pool_evaluators=args.data_pool_evaluators,
                                            dataset_val_evaluators=args.data_val_evaluators,
                                            yoked_learning_only=args.yoked_learning_only,
                                            stop_size=args.stop_size,
                                            stop_cutoff=args.stop_cutoff,
                                            evaluate_stride=args.evaluate_stride,
                                            output_details=args.output_details,
                                            kernel=args.kernel_selector,
                                            save_cpt_stride=args.save_cpt_stride,
                                            seed=args.seed,
                                            logger=args.logger)
                active_learner.run(max_iter=args.max_iter)
                df = pd.read_csv(f'{save_dir}/al_traj.csv')
                # set data selected by active learning to active
                current_selected_id = max([task.selected_id for task in tasks_active]) + 1
                for idxs_add in df['id_add'].apply(lambda x: json.loads(x)):
                    for idx in idxs_add:
                        task = tasks_all.filter_by(id=idx).first()
                        task.active = True
                        task.selected_id = current_selected_id
                    current_selected_id += 1
                # set data not selected by active learning to inactive
                for task in tasks_pool:
                    if not task.active and not task.inactive:
                        task.inactive = True
                session.commit()
                os.remove(f'{save_dir}/kernel_selector.pkl')
            elif margs.learning_type == 'explorative':
                raise NotImplementedError
            elif margs.learning_type == 'exploitive':
                raise NotImplementedError
            else:
                raise NotImplementedError

    @staticmethod
    def create_single_molecule_tasks():
        mols = session.query(Molecule)
        for mol in tqdm(mols, total=mols.count()):
            task = SingleMoleculeTask(molecule_id=mol.id)
            add_or_query(task, ['molecule_id'])
        session.commit()

    @staticmethod
    def create_double_molecule_tasks(group1: str = None, group2: str = None, file: str = None):
        mols = session.query(Molecule)
        if group1 is not None:
            mols1 = mols.filter_by(tag=group1).all()
            if group2 is not None:
                # cross combination of molecules with different tags (e.g. drug and excp).
                mols2 = mols.filter_by(tag=group2).all()
                for mol1 in tqdm(mols1, total=len(mols1)):
                    for mol2 in tqdm(mols2, total=len(mols2)):
                        mid = f'{mol1.id}_{mol2.id}'
                        task = DoubleMoleculeTask(molecules_id=mid)
                        add_or_query(task, ['molecules_id'])
            else:
                # self combination of molecules with the same tag.
                for mol1 in tqdm(mols1, total=len(mols1)):
                    mid = f'{mol1.id}_{mol1.id}'
                    task = DoubleMoleculeTask(molecules_id=mid)
                    add_or_query(task, ['molecules_id'])
        if file is not None:
            df = pd.read_csv(file)
            for i, row in df.iterrows():
                name_mol1, name_mol2 = row.tolist()[:2]
                mol1 = mols.filter_by(name=name_mol1).first()
                mol2 = mols.filter_by(name=name_mol2).first()
                mid = f'{mol1.id}_{mol2.id}'
                task = DoubleMoleculeTask(molecules_id=mid)
                add_or_query(task, ['molecules_id'])
        session.commit()

    def submit_jobs(self, args: MonitorArgs, jobs_to_submit: List, extend: bool = False):
        if jobs_to_submit.count() != 0:
            for job in jobs_to_submit:
                name = job.name + '_extend' if extend else job.name
                commands = json.loads(job.commands_extend) if extend else json.loads(job.commands_mdrun)
                sh = self.job_manager.generate_sh(name=name, path=job.ms_dir,
                                                  partition=args.partition, ntasks=args.ntasks, n_gpu=args.n_gpu,
                                                  memory=args.mem, walltime=args.walltime, exclude=args.exclude,
                                                  commands=commands, save_running_time=True,
                                                  sh_index=True)
                self.job_manager.submit(sh)
                update_list(job, 'sh_file', [sh])
                job.status = Status.SUBMITED
                session.commit()

    def get_jobs_to_analyze(self, table, n_analyze: int) -> List:
        """ get the jobs object to be analyzed.

        Parameters
        ----------
        table: the job table to be analyzed in the
        n_analyze

        Returns
        -------

        """
        if self.job_manager is not None:
            self.job_manager.update_stored_jobs()
        jobs_to_analyze = []
        for job in session.query(table).filter_by(status=Status.SUBMITED):
            if self.job_manager is None or not self.job_manager.is_running(job.slurm_name):
                jobs_to_analyze.append(job)
            if len(jobs_to_analyze) >= n_analyze:
                break
        return jobs_to_analyze

    @staticmethod
    def analyze_multiprocess(analyze_function: Callable, jobs_dirs: List, n_jobs: int = 1) -> List[Dict]:
        """ multiprocess analyze the results.
        Important: the input cannot be  SQLAlchemy objects. They are not allows to be process in a multiprocess manner.

        Parameters
        ----------
        analyze_function: the single-process analyze function.
        jobs_dirs: the directories of jobs.
        n_jobs: the number of multiprocess.

        Returns
        -------
        The analuzed results.
        """
        results = []
        n_analyze = int(math.ceil(len(jobs_dirs) / n_jobs))
        for i in tqdm(range(n_analyze), total=n_analyze):
            jobs_dir = jobs_dirs[i * n_jobs:(i + 1) * n_jobs]
            with Pool(n_jobs) as p:
                result = p.map(analyze_function, jobs_dir)
            results += result
        return results
