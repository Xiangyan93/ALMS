#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import os
import shutil
from .base import GmxSimulation
from ..dff.ppf import delta_ppf


class Npt(GmxSimulation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logs = ['npt.log', 'hvap.log']
        self.n_atom_default = 3000
        self.n_mol_default = 75
        self.dt = 0.002

    def build(self, path: str, smiles_list: List[str],
              n_mol_list: List[int] = None, n_mol_ratio: List[int] = None,
              n_atoms: int = 3000, n_mols: int = 60, length: float = None, density: float = None,
              export: bool = True, ppf: str = None):
        cwd = os.getcwd()
        os.chdir(path)
        n_mol_list, pdb_list, mol2_list, length, box_size, vol = \
            super()._build(path, smiles_list, n_mol_list, n_mol_ratio, n_atoms, n_mols, length, density)
        print('Build coordinates using Packmol: %s molecules ...' % n_mol_list)
        self.packmol.build_box(pdb_list, n_mol_list, self.pdb, box_size=[i - 2 for i in box_size], silent=True)
        print('Create box using DFF ...')
        self.dff.build_box_after_packmol(mol2_list, n_mol_list, self.msd, mol_corr=self.pdb, box_size=box_size)

        # build msd for fast export
        self.packmol.build_box(pdb_list, [1] * len(pdb_list), self._single_pdb, box_size=box_size,
                               inp_file='build_single.inp', silent=True)
        self.dff.build_box_after_packmol(mol2_list, [1] * len(pdb_list), self._single_msd,
                                         mol_corr=self._single_pdb, box_size=box_size)

        if export:
            self.fast_export_single(ppf=ppf, gro_out='_single.gro', top_out='topol.top')
            self.gmx.pdb2gro(self.pdb, 'conf.gro', [i / 10 for i in box_size], silent=True)  # A to nm
            self.gmx.modify_top_mol_numbers('topol.top', n_mol_list)
            if ppf is not None:
                shutil.copy(os.path.join(ppf), 'ff.ppf')

        os.chdir(cwd)

    def prepare(self, path: str, build_dir: str = '../build', n_jobs: int = 1,
                gro: str = 'conf.gro', top: str = 'topol.top', T=298, P=1, TANNEAL=800,
                dt=0.002, nst_eq=int(4E5), nst_run=int(5E5), nst_edr=100, nst_trr=int(5E4), nst_xtc=int(1E3),
                random_seed=-1, drde=False, tcoupl='langevin', T_basic=298) -> List[str]:
        cwd = os.getcwd()
        os.chdir(path)

        if os.path.abspath(build_dir) != os.getcwd():
            shutil.copy(os.path.join(build_dir, gro), gro)
            shutil.copy(os.path.join(build_dir, top), top)
            for f in os.listdir(build_dir):
                if f.endswith('.itp'):
                    shutil.copy(os.path.join(build_dir, f), '.')

        if drde:
            ### Temperature dependent parameters
            # TODO Assumes ppf file named ff.ppf
            if os.path.abspath(build_dir) != os.getcwd():
                shutil.copy(os.path.join(build_dir, self._single_msd), self._single_msd)
            delta_ppf(os.path.join(build_dir, 'ff.ppf'), 'ff.ppf', T, T_basic=T_basic)
            mol_numbers = self.gmx.get_top_mol_numbers(top)
            self.fast_export_single(ppf='ff.ppf', gro_out='_single.gro', top_out=top)
            self.gmx.modify_top_mol_numbers(top, [n for m, n in mol_numbers])

        commands = []
        # energy minimization
        self.gmx.prepare_mdp_from_template('t_em.mdp', mdp_out='grompp-em.mdp')
        cmd = self.gmx.grompp(mdp='grompp-em.mdp', gro=gro, top=top, tpr_out='em.tpr', get_cmd=True)
        commands.append(cmd)
        cmd = self.gmx.mdrun(name='em', nprocs=n_jobs, get_cmd=True)
        commands.append(cmd)

        gro_em = 'em.gro'
        # NVT annealing from 0 to TANNEAL to target T with Langevin thermostat
        if TANNEAL is not None:
            self.gmx.prepare_mdp_from_template('t_nvt_anneal.mdp', mdp_out='grompp-anneal.mdp', T=T, TANNEAL=TANNEAL,
                                               nsteps=int(1E5), nstxtcout=0)
            cmd = self.gmx.grompp(mdp='grompp-anneal.mdp', gro='em.gro', top=top, tpr_out='anneal.tpr', get_cmd=True)
            commands.append(cmd)
            cmd = self.gmx.mdrun(name='anneal', nprocs=n_jobs, get_cmd=True)
            commands.append(cmd)

            gro_em = 'anneal.gro'

        # NPT equilibrium with Langevin thermostat and Berendsen barostat
        self.gmx.prepare_mdp_from_template('t_npt.mdp', mdp_out='grompp-eq.mdp', T=T, P=P, gen_seed=random_seed,
                                           nsteps=nst_eq, nstxtcout=0, pcoupl='berendsen')
        cmd = self.gmx.grompp(mdp='grompp-eq.mdp', gro=gro_em, top=top, tpr_out='eq.tpr', get_cmd=True)
        commands.append(cmd)
        cmd = self.gmx.mdrun(name='eq', nprocs=n_jobs, get_cmd=True)
        commands.append(cmd)

        # NPT production with Langevin thermostat and Parrinello-Rahman barostat
        self.gmx.prepare_mdp_from_template('t_npt.mdp', mdp_out='grompp-npt.mdp', T=T, P=P,
                                           dt=dt, nsteps=nst_run, nstenergy=nst_edr, nstxout=nst_trr, nstvout=nst_trr,
                                           nstxtcout=nst_xtc, restart=True, tcoupl=tcoupl)
        cmd = self.gmx.grompp(mdp='grompp-npt.mdp', gro='eq.gro', top=top, tpr_out='npt.tpr',
                              cpt='eq.cpt', get_cmd=True)
        commands.append(cmd)
        cmd = self.gmx.mdrun(name='npt', nprocs=n_jobs, get_cmd=True)
        commands.append(cmd)

        # Rerun enthalpy of vaporization
        commands.append('export GMX_MAXCONSTRWARN=-1')

        top_hvap = 'topol-hvap.top'
        self.gmx.generate_top_for_hvap(top, top_hvap)
        self.gmx.prepare_mdp_from_template('t_npt.mdp', mdp_out='grompp-hvap.mdp', nstxtcout=0, restart=True)
        cmd = self.gmx.grompp(mdp='grompp-hvap.mdp', gro='eq.gro', top=top_hvap, tpr_out='hvap.tpr', get_cmd=True)
        commands.append(cmd)
        # Use OpenMP instead of MPI when rerun hvap
        cmd = self.gmx.mdrun(name='hvap', nprocs=n_jobs, n_omp=n_jobs, rerun='npt.xtc', get_cmd=True)
        commands.append(cmd)
        
        os.chdir(cwd)
        return commands
