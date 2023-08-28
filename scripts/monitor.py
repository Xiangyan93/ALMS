#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from alms.database.models import *
from alms.args import MonitorArgs, SoftwareArgs
from alms.task.qm_cv import TaskCV
from alms.task.md_npt import TaskNPT
from alms.task.md_binding import TaskBINDING

"""
def active_learning(margs: MonitorArgs):
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
    args.n_jobs = margs.n_jobs
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


def monitor(args: MonitorArgs):
    if args.task == 'qm_cv':
        task = TaskCV(simulator=args.Gaussian, job_manager=args.JobManager)
    elif args.task == 'md_npt':
        task = TaskNPT(force_field=args.ForceField, simulator=args.Simulator, packmol=args.Packmol,
                       job_manager=args.JobManager)
    elif args.task == 'md_binding':
        task = TaskBINDING(force_field=args.ForceField, simulator=args.Simulator, packmol=args.Packmol,
                           job_manager=args.JobManager, plumed=args.Plumed)
    else:
        raise ValueError()

    while True:
        print('Start a new loop\n'
              'Step1: active learning.\n\n')
        task.active_learning(args)
        print('Step2: create.\n\n')
        task.create(args)
        print('\nStep3: build.\n')
        task.build(args)
        """
        print('\nStep4: run.\n')
        task.run(args)
        print('\nStep5: analyze.\n')
        task.analyze(args)
        print('\nStep6: extend.\n')
        task.extend(args)
        print('\nStep7: update failed mols.\n')
        task.update_fail_tasks()
        """
        print('Sleep %d minutes...' % args.t_sleep)
        time.sleep(args.t_sleep * 60)


if __name__ == '__main__':
    monitor(args=MonitorArgs().parse_args())
