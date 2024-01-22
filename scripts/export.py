#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tqdm import tqdm
import pandas as pd
import numpy as np
from alms.args import ExportArgs
from alms.database.models import *
from alms.analysis import *


def export(args: ExportArgs):
    if args.property in ['cp', 'density', 'hvap']:
        tasks = session.query(SingleMoleculeTask)
    elif args.property in ['binding_free_energy']:
        tasks = session.query(DoubleMoleculeTask)
    else:
        raise ValueError(f'Invalid property: {args.property}')
    mols = session.query(Molecule)

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
        for task in tqdm(tasks, total=tasks.count()):
            results = get_cp(task)
            if results is None:
                continue
            T_list, P_list, cp, cp_inter, cp_intra, cp_pv = results
            # update dataframe
            d['smiles'] += [task.molecule.smiles] * len(T_list)
            d['T'] += T_list
            d['P'] += P_list
            d['cp'] += cp
            d['cp_inter'] += cp_inter.tolist()
            d['cp_intra'] += cp_intra.tolist()
            d['cp_pv'] += cp_pv.tolist()
            d['red_T'] += (np.asarray(T_list) / task.molecule.tc).tolist()
        pd.DataFrame(d).to_csv('cp.csv', index=False)
    elif args.property == 'density':
        d = {
            'smiles': [],
            'T': [],
            'P': [],
            'density': [],
            'red_T': []
        }
        for task in tqdm(tasks, total=tasks.count()):
            results = get_density(task)
            if results is None:
                continue
            T_list, P_list, density = results
            # update dataframe
            d['smiles'] += [task.molecule.smiles] * len(T_list)
            d['T'] += T_list
            d['P'] += P_list
            d['density'] += density
            d['red_T'] += (np.asarray(T_list) / task.molecule.tc).tolist()
        pd.DataFrame(d).to_csv('density.csv', index=False)
    elif args.property == 'hvap':
        d = {
            'smiles': [],
            'T': [],
            'P': [],
            'hvap': [],
            'red_T': []
        }
        for task in tqdm(tasks, total=tasks.count()):
            results = get_hvap(task)
            if results is None:
                continue
            T_list, P_list, hvap = results
            # update dataframe
            d['smiles'] += [task.molecule.smiles] * len(T_list)
            d['T'] += T_list
            d['P'] += P_list
            d['hvap'] += hvap
            d['red_T'] += (np.asarray(T_list) / task.molecule.tc).tolist()
        pd.DataFrame(d).to_csv('hvap.csv', index=False)
    elif args.property == 'binding_free_energy':
        df_train = pd.DataFrame({'smiles1': [], 'name1': [], 'smiles2': [], 'name2': [], 'binding_free_energy': []})
        df_test = pd.DataFrame({'smiles1': [], 'name1': [], 'smiles2': [], 'name2': [], 'binding_free_energy': []})
        for task in session.query(DoubleMoleculeTask):
            mol1 = task.molecule_1
            mol2 = task.molecule_2
            if mol2.tag == 'drug' and mol1.tag == 'excp':
                mol1, mol2 = mol2, mol1
            if task.properties is None or 'binding_free_energy' not in json.loads(task.properties):
                df_test.loc[len(df_test)] = [mol1.smiles, mol1.name, mol2.smiles, mol2.name, 0.]
            else:
                df_train.loc[len(df_train)] = [mol1.smiles, mol1.name, mol2.smiles, mol2.name,
                                               json.loads(task.properties)['binding_free_energy']]
        df_train.to_csv('binding_free_energy_train.csv', index=False)
        df_test.to_csv('binding_free_energy_test.csv', index=False)
    elif args.property is None:
        smiles = [task.molecule.smiles for task in session.query(SingleMoleculeTask)]
        al = [task.active for task in session.query(SingleMoleculeTask)]
        pd.DataFrame({'smiles': smiles, 'active_learning': al}).to_csv('molecules.csv', index=False)


if __name__ == '__main__':
    export(args=ExportArgs().parse_args())
