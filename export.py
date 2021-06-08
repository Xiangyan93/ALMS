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
            # intermolecular cp.
            jobs = [job for job in mol.md_npt if job.status == Status.ANALYZED]
            if len(jobs) < 5:
                continue
            n_mols = [json.loads(job.result)['n_mols'] for job in jobs]
            assert len(set(n_mols)) == 1
            einter = [json.loads(job.result)['einter'][0] / n_mols[0] for job in jobs]
            T_list = [job.T for job in jobs]
            P_list = [job.P for job in jobs]
            cp_inter = get_cp_inter(T_list, P_list, einter)
            if cp_inter is None:
                continue
            # intramolecular cp.
            jobs = [job for job in mol.qm_cv if job.status == Status.ANALYZED]
            if len(jobs) == 0:
                continue
            T_list_in = json.loads(jobs[0].result)['T']
            CV_list = json.loads(jobs[0].result)['cv_corrected']
            cp_intra = get_cp_intra(T_list_in=T_list_in, CV_list=CV_list, T_list_out=T_list)
            # update dataframe
            d['smiles'] += [mol.smiles] * len(T_list)
            d['T'] += T_list
            d['P'] += P_list
            #if not is_monotonic((cp_inter + cp_intra).tolist()):
            # print(mol.smiles)
                #print(cp_intra, cp_inter)
                #print((cp_inter + cp_intra).tolist())
                # exit()
            d['cp'] += (cp_inter + cp_intra).tolist()
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
