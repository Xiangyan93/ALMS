#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tqdm import tqdm
from alms.database.models import *
from .base import BaseTask
from ..args import MonitorArgs


class TaskAL(BaseTask):
    def active_learning(self, margs: MonitorArgs):
        if margs.strategy == 'all':
            mols_all = session.query(SingleMoleculeTask)
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