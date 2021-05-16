#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tqdm import tqdm
from aims.args import ActiveLearningArgs as ALArgs
from aims.database import *
from aims.ml.mgk.args import ActiveLearningArgs
from aims.ml.mgk.data.data import Dataset
from aims.ml.mgk.kernels.utils import get_kernel_config
from aims.ml.mgk.evaluator import ActiveLearner


def active_learning(alargs: ALArgs):
    args = ActiveLearningArgs()
    args.save_dir = None
    args.n_jobs = alargs.n_jobs
    args.pure_columns = ['smiles']
    args.target_columns = ['target']
    args.graph_kernel_type = 'graph'
    args.graph_hyperparameters = ['data/tMGR.json']
    args.model_type = 'gpr'
    args.alpha = 0.01
    args.learning_algorithm = 'unsupervised'
    args.add_size = 1
    args.cluster_size = 1
    args.stop_size = 100000
    args.stop_uncertainty = alargs.stop_uncertainty
    args.evaluate_stride = 100000

    mols_all = session.query(Molecule)
    # set all molecules with active_learning = False.
    if alargs.rerun:
        for mol in tqdm(mols_all.all(), total=mols_all.count()):
            mol.active_learning = False
        session.commit()
    # random select 2 samples as the start of active learning.
    if mols_all.filter_by(active_learning=True).count() <= 1:
        for mol in np.random.choice(mols_all.all(), 2, replace=False):
            mol.active_learning = True
        session.commit()

    # get selected data set.
    mols = mols_all.filter_by(active_learning=True)
    df = pd.DataFrame({'smiles': [mol.smiles for mol in mols],
                       'target': [0.] * mols.count()})
    dataset = Dataset.from_df(args, df)
    dataset.update_args(args)
    # get pool data set.
    mols = mols_all.filter_by(active_learning=False)
    df_pool = pd.DataFrame({'smiles': [mol.smiles for mol in mols],
                            'target': [0.] * mols.count()})
    dataset_pool = Dataset.from_df(args, df_pool)
    dataset_pool.update_args(args)
    # get full data set.
    dataset_full = dataset.copy()
    dataset_full.data = dataset.data + dataset_pool.data
    dataset_full.unify_datatype()
    # get preCalc kernel
    print('**\tCalculating kernel matrix\t**')
    kernel_config = get_kernel_config(args, dataset_full)
    print('**\tEnd Calculating kernel matrix\t**')
    kernel_config = kernel_config.get_preCalc_kernel_config(args, dataset_full)
    dataset.graph_kernel_type = 'preCalc'
    dataset_pool.graph_kernel_type = 'preCalc'
    dataset_full.graph_kernel_type = 'preCalc'
    # Active Learning
    al = ActiveLearner(args, dataset, dataset_pool, kernel_config)
    al.run()
    smiles = [s.split(',')[0] for s in al.dataset.X_repr.ravel()]
    for mol in tqdm(mols_all.all(), total=mols_all.count()):
        if mol.smiles in smiles:
            mol.active_learning = True
    session.commit()


if __name__ == '__main__':
    active_learning(alargs=ALArgs().parse_args())
