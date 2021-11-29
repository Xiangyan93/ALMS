#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from aims.database import *
from aims.args import MonitorArgs
from aims.ml.mgk.args import ActiveLearningArgs
from aims.ml.mgk.kernels.utils import get_kernel_config
from aims.ml.mgk.data.data import Dataset
from aims.ml.mgk.evaluator import ActiveLearner


def active_learning(margs: MonitorArgs):
    if margs.stop_uncertainty is None:
        return
    if margs.stop_uncertainty < 0.0:
        mols_all = session.query(Molecule)
        for mol in tqdm(mols_all.all(), total=mols_all.count()):
            mol.active = True
            mol.inactive = False
        session.commit()
        return
    args = ActiveLearningArgs()
    args.dataset_type = 'regression'
    args.save_dir = 'data'
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
    args.evaluate_stride = 100000
    args.seed = margs.seed

    mols_all = session.query(Molecule).filter_by(fail=False)
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
    smiles = [s.split(',')[0] for s in al.dataset.X_repr.ravel()]
    for mol in tqdm(mols_all.all(), total=mols_all.count()):
        if mol.smiles in smiles:
            mol.active = True
            mol.inactive = False
    smiles = [s.split(',')[0] for s in al.dataset_pool.X_repr.ravel()]
    for mol in tqdm(mols_all.all(), total=mols_all.count()):
        if mol.smiles in smiles:
            mol.active = False
            mol.inactive = True
    session.commit()


def monitor(args: MonitorArgs):
    if args.task == 'qm_cv':
        from aims.qm.qm_cv import get_GaussianSimulator, create, build, run, analyze, extend
        from aims.analysis.cp import update_fail_mols
        simulator = get_GaussianSimulator(args)
    elif args.task == 'md_npt':
        from aims.md.md_npt import get_NptSimulator, create, build, run, analyze, extend
        from aims.analysis.cp import update_fail_mols
        simulator = get_NptSimulator(args)
    else:
        return
    job_manager = args.job_manager_

    while True:
        print('Start a new loop\n'
              'Step1: active learning.\n\n')
        active_learning(args)
        print('Step2: create.\n\n')
        create(args)
        print('\nStep3: build.\n')
        build(args, simulator)
        print('\nStep4: run.\n')
        run(args, simulator, job_manager)
        print('\nStep5: analyze.\n')
        analyze(args, simulator, job_manager)
        print('\nStep6: extend.\n')
        extend(args, simulator, job_manager)
        print('\nStep7: update failed mols.\n')
        update_fail_mols()
        print('Sleep %d minutes...' % args.t_sleep)
        time.sleep(args.t_sleep * 60)


if __name__ == '__main__':
    monitor(args=MonitorArgs().parse_args())
