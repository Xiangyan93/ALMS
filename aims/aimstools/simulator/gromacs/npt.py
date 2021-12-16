#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import os
import numpy as np
import shutil
import physical_validation as pv
from panedr.panedr import edr_to_df
from .base import GmxSimulation
from ..dff.ppf import delta_ppf
from ...utils import is_converged, block_average


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

    # analyze thermodynamic properties
    def analyze(self, path: str, check_converge: bool = True, cutoff_time: int = 7777) -> Dict:
        cwd = os.getcwd()
        os.chdir(path)

        info_dict = dict()
        try:
            npt_edr = 'npt.edr'
            hvap_edr = 'hvap.edr'

            df = edr_to_df(npt_edr)
            df_hvap = edr_to_df(hvap_edr)
            potential_series = df.Potential
            density_series = df.Density
            T_sim = df.Temperature.mean()
            time_sim = df.Potential.index[-1]  # unit: ps
            einter_series = df_hvap.Potential
            info_dict['time_sim'] = time_sim
            ### Check the ensemble. KS test on the distribution of kinetic energy
            ### TODO This can be optimized
            data = pv.data.GromacsParser(self.gmx.GMX_EXE).get_simulation_data(
                mdp='grompp-npt.mdp', top='topol.top', edr=npt_edr)
            p = pv.kinetic_energy.distribution(data, strict=True, verbosity=0)
            # If test does not pass, set the desired temperature to t_real.
            # Because small deviation in temperature exists for Langevin thermostat
            # 3 Kelvin of deviation is permitted
            if p < 0.01 and abs(data.ensemble.temperature - T_sim) < 3:
                try:
                    data._SimulationData__ensemble._EnsembleData__t = T_sim
                    p = pv.kinetic_energy.distribution(data, strict=True, verbosity=0)
                except Exception as e:
                    print(repr(e))

            if p < 0.01:
                if time_sim > cutoff_time:
                    info_dict['warning'] = 'KS test for kinetic energy failed: p<0.01'
                else:
                    info_dict['failed'] = False
                    info_dict['continue'] = True
                    info_dict['continue_n'] = 2.5e5
                    info_dict['reason'] = 'KS test for kinetic energy failed: p<0.01'
                    return info_dict
            elif p < 0.05:
                if time_sim > cutoff_time:
                    info_dict['warning'] = 'KS test for kinetic energy failed: 0.01 < p < 0.05'
                else:
                    info_dict['failed'] = False
                    info_dict['continue'] = True
                    info_dict['continue_n'] = 2.5e5
                    info_dict['reason'] = 'KS test for kinetic energy failed: 0.01 < p < 0.05'
                    return info_dict

            ### Check structure freezing using Density
            if density_series.min() / 1000 < 0.1:  # g/mL
                if time_sim > 2000.0:
                    info_dict['failed'] = True
                    info_dict['reason'] = 'vaporize'
                    return info_dict
                else:
                    info_dict['failed'] = False
                    info_dict['continue'] = True
                    info_dict['continue_n'] = 2.5e5
                    info_dict['reason'] = 'vaporize'
                    return info_dict

            ### Check structure freezing using Diffusion of COM of molecules. Only use last 400 ps data
            diffusion, _ = self.gmx.diffusion('npt.xtc', 'npt.tpr', mol=True, begin=time_sim - 400)
            if diffusion < 1E-8:  # cm^2/s
                if time_sim > 2000.0:
                    info_dict['failed'] = True
                    info_dict['reason'] = 'freeze'
                    return info_dict
                else:
                    info_dict['failed'] = False
                    info_dict['continue'] = True
                    info_dict['continue_n'] = 2.5e5
                    info_dict['reason'] = 'freeze'
                    return info_dict

            ### Check convergence
            if check_converge:
                # use potential to do a initial determination
                # use at least 4/5 of the data
                _, when_pe = is_converged(potential_series, frac_min=0)
                when_pe = min(when_pe, time_sim * 0.2)
                # use density to do a final determination
                _, when_dens = is_converged(density_series, frac_min=0)
                when = max(when_pe, when_dens)
                if when > time_sim * 0.5:
                    if time_sim > cutoff_time:
                        info_dict['warning'] = 'PE and density not converged'
                    else:
                        info_dict['failed'] = False
                        info_dict['continue'] = True
                        info_dict['continue_n'] = 2.5e5
                        info_dict['reason'] = 'PE and density not converged'
                        return info_dict
            else:
                when = 0

            ### Get expansion and compressibility using fluctuation method
            nblock = 5
            blocksize = (time_sim - when) / nblock
            expan_list = []
            compr_list = []
            for i in range(nblock):
                begin = when + blocksize * i
                end = when + blocksize * (i + 1)
                expan, compr = self.gmx.get_fluct_props('npt.edr', begin=begin, end=end)
                expan_list.append(expan)
                compr_list.append(compr)
            expansion, expan_stderr = np.mean(expan_list), np.std(expan_list, ddof=1) / np.sqrt(nblock)
            compressi, compr_stderr = np.mean(compr_list), np.std(compr_list, ddof=1) / np.sqrt(nblock)
            expan_stderr = float('%.1e' % expan_stderr)  # 2 effective number for stderr
            compr_stderr = float('%.1e' % compr_stderr)  # 2 effective number for stderr

            temperature_and_stderr, pressure_and_stderr, potential_and_stderr, density_and_stderr, volume_and_stderr, \
            ke_and_stderr, te_and_stderr, pv_and_stderr = \
                self.gmx.get_properties_stderr(npt_edr,
                                               ['Temperature', 'Pressure', 'Potential', 'Density', 'Volume', 'Kinetic-En.',
                                                'Total-Energy', 'pV'], begin=when)
            if info_dict.get('failed') is None:
                info_dict['failed'] = False
            if info_dict.get('continue') is None:
                info_dict['continue'] = False
            if info_dict.get('reason') is None:
                info_dict['reason'] = 'converge'

            le_and_stderr = []
            le_and_stderr.append(te_and_stderr[0] + pv_and_stderr[0])
            le_and_stderr.append(te_and_stderr[1] + pv_and_stderr[1])
            ad_dict = {
                'n_mols': self.gmx.get_n_mols_from_top('topol.top'),
                'density': [i / 1000 for i in density_and_stderr],  # g/mL
                'length': time_sim,
                'converge': when,
                'temperature': temperature_and_stderr,  # K
                'pressure': pressure_and_stderr,  # bar
                'potential': potential_and_stderr,  # kJ/mol
                'kinetic energy': ke_and_stderr,  # kJ/mol
                'total energy': te_and_stderr,  # kJ/mol
                'pV': pv_and_stderr,  # kJ/mol
                'liquid enthalpy': le_and_stderr,  # kJ/mol
                'einter': list(block_average(einter_series.loc[when:])),  # kJ/mol
                'expansion': [expansion, expan_stderr],  # 1/K
                'compress': [compressi, compr_stderr],  # m^3/J
            }
            info_dict.update(ad_dict)
        except:
            info_dict['failed'] = True
            info_dict['reason'] = 'Abnormal Error.'
        os.chdir(cwd)
        return info_dict

    def extend(self, path: str, continue_n: int, dt: float = 0.002, n_srun: int = 1, n_tomp: int = 1) -> List[str]:
        cwd = os.getcwd()
        os.chdir(path)

        commands = []
        extend = continue_n * dt
        self.gmx.extend_tpr('npt.tpr', extend, silent=True)
        # Extending NPT production
        cmd = self.gmx.mdrun(name='npt', nprocs=n_srun, n_omp=n_tomp, extend=True, get_cmd=True)
        commands.append(cmd)

        # Rerun enthalpy of vaporization
        commands.append('export GMX_MAXCONSTRWARN=-1')
        # Use OpenMP instead of MPI when rerun hvap
        cmd = self.gmx.mdrun(name='hvap', nprocs=1, n_omp=n_srun * n_tomp, rerun='npt.xtc', get_cmd=True)
        commands.append(cmd)

        os.chdir(cwd)
        return commands
