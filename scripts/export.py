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
    elif args.property in ['binding_fe']:
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
    elif args.property == 'binding_fe':
        df = pd.DataFrame({'drug_smiles': [], 'drug_name': [], 'excp_smiles': [], 'excp_name': [], 'binding_fe': []})
        for task in session.query(DoubleMoleculeTask):
            mol1 = task.molecule_1
            mol2 = task.molecule_2
            binding_fe_de = []
            for job in task.md_binding:
                fe = json.loads(job.result).get('binding_free_energy')
                if fe:
                    binding_fe_de.append(fe)
            from scipy import stats
            z_scores = np.abs(stats.zscore(binding_fe_de))
            if max(z_scores) > 3:
                print(z_scores, binding_fe_de)
            if len(binding_fe_de) == 0:
                continue
            assert len(binding_fe_de) >= 5
            df.loc[len(df)] = [mol1.smiles, mol1.name, mol2.smiles, mol2.name,
                               np.mean(binding_fe_de)]
        df.to_csv('binding_fe.csv', index=False)
    elif args.property is None:
        smiles = [task.molecule.smiles for task in session.query(SingleMoleculeTask)]
        al = [task.active for task in session.query(SingleMoleculeTask)]
        pd.DataFrame({'smiles': smiles, 'active_learning': al}).to_csv('molecules.csv', index=False)


if __name__ == '__main__':
    export(args=ExportArgs().parse_args())
