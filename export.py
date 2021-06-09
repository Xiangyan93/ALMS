#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tqdm import tqdm
import pandas as pd
from aims.args import ExportArgs
from aims.database.models import *
from aims.analysis.cp import *


def export(args: ExportArgs):
    if args.property == 'cp':
        d = {
            'smiles': [],
            'T': [],
            'P': [],
            'cp_inter': [],
            'cp_intra': [],
            'cp': [],
            'red_T': []
        }
        mols = session.query(Molecule)# .filter_by(smiles='CC1=C(C)C(=O)C(C)=C(C)C1=O')
        for mol in tqdm(mols, total=mols.count()):
            results = get_cp(mol)
            if results is None:
                continue
            T_list, P_list, cp, cp_inter, cp_intra = results
            # update dataframe
            d['smiles'] += [mol.smiles] * len(T_list)
            d['T'] += T_list
            d['P'] += P_list
            #if not is_monotonic((cp_inter + cp_intra).tolist()):
            # print(mol.smiles)
                #print(cp_intra, cp_inter)
                #print((cp_inter + cp_intra).tolist())
                # exit()
            d['cp'] += cp
            d['cp_inter'] += cp_inter.tolist()
            d['cp_intra'] += cp_intra.tolist()
            d['red_T'] += (np.asarray(T_list) / mol.tc).tolist()
        pd.DataFrame(d).to_csv('cp.csv', index=False)
    elif args.property == 'density':
        d = {
            'smiles': [],
            'T': [],
            'P': [],
            'density': [],
            'density_u': [],
            'red_T': []
        }
        jobs = session.query(MD_NPT).filter_by(status=Status.ANALYZED)
        for job in tqdm(jobs, total=jobs.count()):
            result = json.loads(job.result)
            d['smiles'].append(job.molecule.smiles)
            d['T'].append(job.T)
            d['P'].append(job.P)
            d['density'].append(result['density'][0] * 1000)  # kg/m3
            d['density_u'].append(result['density'][1] * 1000)
            d['red_T'].append(job.T / job.molecule.tc)
        pd.DataFrame(d).to_csv('density.csv', index=False)
    elif args.property is None:
        smiles = [mol.smiles for mol in session.query(Molecule)]
        al = [mol.active_learning for mol in session.query(Molecule)]
        pd.DataFrame({'smiles': smiles, 'active_learning': al}).to_csv('molecules.csv', index=False)


if __name__ == '__main__':
    export(args=ExportArgs().parse_args())
