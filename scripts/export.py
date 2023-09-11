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
        df = pd.DataFrame({'drug_smiles': [], 'drug_name': [], 'excp_smiles': [], 'excp_name': [], 'binding_fe_de': [],
                           'binding_fe_dd': [], 'binding_fe_ee': []})
        for mol1 in mols:
            if mol1.tag != 'drug' or mol1.id != 7:
                continue
            for mol2 in mols:
                if mol2.tag != 'excp' or mol2.id in [81, 87, 94]:
                    continue
                task = session.query(DoubleMoleculeTask).filter_by(molecules_id=f'{mol1.id}_{mol2.id}').first()
                binding_fe_de = []
                for job in task.md_binding:
                    fe = json.loads(job.result).get('binding_free_energy')
                    if fe and fe > -50:
                        binding_fe_de.append(fe)
                if np.min(binding_fe_de) < -40:
                    print(mol1.id, mol2.id, binding_fe_de)
                assert len(binding_fe_de) >= 4

                task = session.query(DoubleMoleculeTask).filter_by(molecules_id=f'{mol1.id}_{mol1.id}').first()
                binding_fe_dd = []
                for job in task.md_binding:
                    fe = json.loads(job.result).get('binding_free_energy')
                    if fe and fe > -50:
                        binding_fe_dd.append(fe)
                if np.min(binding_fe_dd) < -40:
                    print(mol1.id, mol1.id, binding_fe_dd)
                assert len(binding_fe_dd) >= 4

                task = session.query(DoubleMoleculeTask).filter_by(molecules_id=f'{mol2.id}_{mol2.id}').first()
                binding_fe_ee = []
                for job in task.md_binding:
                    fe = json.loads(job.result).get('binding_free_energy')
                    if fe and fe > -50:
                        binding_fe_ee.append(fe)
                if np.min(binding_fe_ee) < -40:
                    print(mol2.id, mol2.id, binding_fe_ee)
                assert len(binding_fe_ee) >= 4
                print(binding_fe_dd, np.mean(binding_fe_dd))
                df.loc[len(df)] = [mol1.smiles, mol1.name, mol2.smiles, mol2.name,
                                   np.mean(binding_fe_de), np.mean(binding_fe_dd), np.mean(binding_fe_ee)]
        df.to_csv('binding_fe.csv', index=False)
    elif args.property is None:
        smiles = [task.molecule.smiles for task in session.query(SingleMoleculeTask)]
        al = [task.active for task in session.query(SingleMoleculeTask)]
        pd.DataFrame({'smiles': smiles, 'active_learning': al}).to_csv('molecules.csv', index=False)


if __name__ == '__main__':
    export(args=ExportArgs().parse_args())
