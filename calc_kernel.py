#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
import pandas as pd
from alms.args import KernelArgs as KArgs
from alms.database import *
from alms.ml.mgk.args import KernelArgs
from alms.ml.mgk.data.data import Dataset
from alms.ml.mgk.kernels.utils import get_kernel_config


def calc_kernel(kargs: KArgs):
    args = KernelArgs()
    args.n_jobs = kargs.n_jobs
    args.save_dir = 'data'
    args.graph_kernel_type = 'graph'
    args.pure_columns = ['smiles']
    args.target_columns = ['target']
    args.graph_hyperparameters = ['data/tMGR.json']

    # get dataset.
    mols = session.query(Molecule)
    df = pd.DataFrame({'smiles': [mol.smiles for mol in mols],
                       'target': [0.] * mols.count()})
    dataset = Dataset.from_df(args, df)
    dataset.update_args(args)
    # set kernel_config
    kernel_config = get_kernel_config(args, dataset)
    print('**\tCalculating kernel matrix\t**')
    kernel_dict = kernel_config.get_kernel_dict(dataset.X, dataset.X_repr.ravel())
    print('**\tEnd Calculating kernel matrix\t**')
    kernel_pkl = os.path.join(args.save_dir, 'kernel.pkl')
    pickle.dump(kernel_dict, open(kernel_pkl, 'wb'), protocol=4)


if __name__ == '__main__':
    calc_kernel(kargs=KArgs().parse_args())
