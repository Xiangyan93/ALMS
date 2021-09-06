#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tqdm import tqdm
import pandas as pd
import numpy as np
from aims.args import ExportArgs
from aims.database.models import *
from aims.analysis import *


def export(args: ExportArgs):
    if args.use_all:
        mols = session.query(Molecule)
    else:
        mols = session.query(Molecule).filter_by(active_learning=True)

    if args.property == 'cp':
        d = {
            'smiles': [],
            'T': [],
            'P': [],
            'cp_inter': [],
            'cp_intra': [],
            'cp_pv': [],
            'cp': [],
            'red_T': []
        }
        for mol in tqdm(mols, total=mols.count()):
            results = get_cp(mol)
            if results is None:
                continue
            T_list, P_list, cp, cp_inter, cp_intra, cp_pv = results
            # update dataframe
            d['smiles'] += [mol.smiles] * len(T_list)
            d['T'] += T_list
            d['P'] += P_list
            d['cp'] += cp
            d['cp_inter'] += cp_inter.tolist()
            d['cp_intra'] += cp_intra.tolist()
            d['cp_pv'] += cp_pv.tolist()
            d['red_T'] += (np.asarray(T_list) / mol.tc).tolist()
        pd.DataFrame(d).to_csv('cp.csv', index=False)
    elif args.property == 'density':
        d = {
            'smiles': [],
            'T': [],
            'P': [],
            'density': [],
            'red_T': []
        }
        for mol in tqdm(mols, total=mols.count()):
            results = get_density(mol)
            if results is None:
                continue
            T_list, P_list, density = results
            # update dataframe
            d['smiles'] += [mol.smiles] * len(T_list)
            d['T'] += T_list
            d['P'] += P_list
            d['density'] += density
            d['red_T'] += (np.asarray(T_list) / mol.tc).tolist()
        pd.DataFrame(d).to_csv('density.csv', index=False)
    elif args.property == 'hvap':
        d = {
            'smiles': [],
            'T': [],
            'P': [],
            'hvap': [],
            'red_T': []
        }
        for mol in tqdm(mols, total=mols.count()):
            results = get_hvap(mol)
            if results is None:
                continue
            T_list, P_list, hvap = results
            # update dataframe
            d['smiles'] += [mol.smiles] * len(T_list)
            d['T'] += T_list
            d['P'] += P_list
            d['hvap'] += hvap
            d['red_T'] += (np.asarray(T_list) / mol.tc).tolist()
        pd.DataFrame(d).to_csv('hvap.csv', index=False)
    elif args.property is None:
        smiles = [mol.smiles for mol in session.query(Molecule)]
        al = [mol.active_learning for mol in session.query(Molecule)]
        pd.DataFrame({'smiles': smiles, 'active_learning': al}).to_csv('molecules.csv', index=False)


if __name__ == '__main__':
    export(args=ExportArgs().parse_args())
