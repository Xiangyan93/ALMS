#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
from simutools.utils.rdkit import mol_filter
from simutools.utils.utils import random_string
from alms.args import SubmitArgs
from alms.database.models import *
from chemprop.args import PredictArgs
from chemprop.train import make_predictions


def submit(args: SubmitArgs):
    for file in args.files:
        df = pd.read_csv(file)
        for i, row in tqdm(df.iterrows(), total=len(df)):
            smiles = mol_filter(smiles=row['smiles'],
                                excluded_smarts=args.excluded_smarts,
                                heavy_atoms=args.heavy_atoms)
            name = row.get('name') or random_string(8)
            if smiles is not None:
                resname = random_string(3)
                mol = Molecule(smiles=smiles, name=name, resname=resname, tag=args.tag)
                add_or_query(mol, ['smiles'])
    session.commit()


def predict(target_property: str):
    mols = session.query(Molecule)
    smiles = []
    for mol in mols:
        smiles.append([mol.smiles])
    args = PredictArgs()
    args.test_path = 'none'
    args.preds_path = f'../data/tmp/{target_property}.csv'
    args.checkpoint_dir = '../ml-models/%s' % target_property
    args.features_generator = ['rdkit_2d_normalized']
    args.no_features_scaling = True
    args.process_args()
    preds = make_predictions(args, smiles)
    for i, mol in enumerate(mols):
        update_dict(mol, 'property_ml', {target_property: preds[i][0]})
    session.commit()


if __name__ == '__main__':
    submit(args=SubmitArgs().parse_args())
    # predict('tt')
    # predict('tb')
    # predict('tc')
