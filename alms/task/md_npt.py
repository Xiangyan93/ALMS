#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import math
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
from sqlalchemy.sql import or_
from rdkit import Chem
import physical_validation as pv
from panedr.panedr import edr_to_df
from simutools.forcefields.amber import AMBER
from simutools.simulator.program import Packmol, GROMACS
from simutools.simulator.func import build
from simutools.utils.series import is_converged, block_average
from .base import BaseTask
from ..args import MonitorArgs
from ..database.models import *
from ..analysis.cp import update_fail_mols


class TaskNPT(BaseTask):
    def __init__(self, job_manager: Slurm, force_field: Union[AMBER], simulator: Union[GROMACS], packmol: Packmol):
        super().__init__(job_manager=job_manager)
        self.ff = force_field
        self.simulator = simulator
        self.packmol = packmol

    def active_learning(self, margs: MonitorArgs):
        self.create_single_molecule_tasks()
        super().active_learning(margs)

    def create(self, args: MonitorArgs):
        tasks = session.query(SingleMoleculeTask).filter(SingleMoleculeTask.active == True)
        for task in tqdm(tasks, total=tasks.count()):
            task.create_jobs(task='md_npt', T_min=args.T_range[0], T_max=args.T_range[1],
                             n_T=args.n_Temp, P_list=args.P_list)
        session.commit()

    def build(self, args: MonitorArgs):
        cwd = os.getcwd()
        # pick args.n_prepare tasks.
        tasks = []
        for task in session.query(SingleMoleculeTask).filter(SingleMoleculeTask.active == True):
            if len(task.status('md_npt')) == 1 and task.status('md_npt')[0] == Status.STARTED:
                tasks.append(task)
            if len(tasks) == args.n_prepare:
                break
        # checkout force field parameters for the molecules.
        for task in tqdm(tasks, total=len(tasks)):
            task.molecule.checkout(force_field=self.ff, simulator=self.simulator)
        # Job.status: STARTED -> BUILD
        # # build initial simulation box and checkout the force field parameters.
        for task in tqdm(tasks, total=len(tasks)):
            mol = task.molecule
            # create build dir. All simulations start from the same initial structure.
            path = os.path.join(task.ms_dir, 'md_npt', 'build')
            create_missing_folders(path)
            os.chdir(path)
            n_molecules = int(3000 / mol.GetNumAtoms)
            V = 10 / 6.022 * mol.molwt / mol.estimate_density  # nm^3
            length = V ** (1 / 3)  # assume cubic box
            self.packmol.build_uniform(pdb_files=[f'{mol.ms_dir}/{mol.name}_ob.pdb'],
                                       n_mol_list=[n_molecules],
                                       output='initial.pdb', box_size=[length] * 3)
            self.simulator.convert_pdb(pdb='initial.pdb', tag_out='initial', box_size=[length] * 3)
            self.simulator.modify_top_mol_numbers(top=f'{mol.ms_dir}/topol.top', outtop='topol.top',
                                                  mol_name=mol.resname, n_mol=n_molecules)
            for job in task.md_npt:
                job.status = Status.BUILD
            session.commit()
        # Job.status: BUILD -> PREPARED
        # # prepares files for the jobs. All commands are saved and then submitted to SLURM.
        jobs = session.query(MD_NPT).filter_by(status=Status.BUILD)
        for job in tqdm(jobs, total=jobs.count()):
            commands = []
            if isinstance(self.simulator, GROMACS):
                os.chdir(job.ms_dir)
                shutil.copy('../build/initial.gro', '.')
                shutil.copy('../build/topol.top', '.')
                gmx = self.simulator
                # energy minimization
                gmx.generate_mdp_from_template(template='t_em.mdp', mdp_out='em.mdp')
                commands += [gmx.grompp(mdp='em.mdp', gro='initial.gro', top='topol.top', tpr='em.tpr',
                                        maxwarn=2, exe=False),
                             gmx.mdrun(tpr='em.tpr', ntomp=args.ntasks, exe=False)]
                # NVT annealing from 0 to TA_annealing to target T with Langevin thermostat
                gmx.generate_mdp_from_template(template='t_nvt_anneal.mdp', mdp_out='anneal.mdp',
                                               T=job.T, T_annealing=800, nsteps=int(1E5), nstxtcout=0)
                commands += [gmx.grompp(mdp='anneal.mdp', gro='em.gro', top='topol.top', tpr='anneal.tpr', exe=False),
                             gmx.mdrun(tpr='anneal.tpr', ntomp=args.ntasks, exe=False)]
                # NPT equilibrium with Langevin thermostat and Berendsen barostat
                gmx.generate_mdp_from_template(template='t_npt.mdp', mdp_out='eq.mdp', T=job.T, P=job.P,
                                               nsteps=int(4E5), nstxtcout=0, tcoupl='langevin', pcoupl='berendsen')
                commands += [gmx.grompp(mdp='eq.mdp', gro='anneal.gro', top='topol.top', tpr='eq.tpr', exe=False),
                             gmx.mdrun(tpr='eq.tpr', ntomp=args.ntasks, exe=False)]
                # NPT production with Langevin thermostat and Parrinello-Rahman barostat
                gmx.generate_mdp_from_template(template='t_npt.mdp', mdp_out='npt.mdp', T=job.T, P=job.P,
                                               dt=0.002, nsteps=int(5E5), nstenergy=100,
                                               nstxout=int(5E4), nstvout=int(5E4),
                                               nstxtcout=1000, restart=True,
                                               tcoupl='langevin', pcoupl='parrinello-rahman')
                commands += [gmx.grompp(mdp='npt.mdp', gro='eq.gro', top='topol.top', tpr='npt.tpr', exe=False),
                             gmx.mdrun(tpr='npt.tpr', ntomp=args.ntasks, exe=False)]
                # Rerun enthalpy of vaporization
                commands.append('export GMX_MAXCONSTRWARN=-1')

                gmx.generate_top_for_hvap(top='topol.top', top_out='topol-hvap.top')
                gmx.generate_mdp_from_template(template='t_npt.mdp', mdp_out='hvap.mdp', nstxtcout=0, restart=True)
                commands += [gmx.grompp(mdp='hvap.mdp', gro='eq.gro', top='topol-hvap.top', tpr='hvap.tpr', exe=False),
                             gmx.mdrun(tpr='hvap.tpr', ntomp=args.ntasks, rerun='npt.xtc', exe=False)]
            else:
                raise ValueError
            job.commands_mdrun = json.dumps(commands)
            job.status = Status.PREPARED
            session.commit()
        os.chdir(cwd)

    def run(self, args: MonitorArgs):
        n_submit = args.n_run - self.job_manager.n_current_jobs
        if n_submit > 0:
            jobs_to_submit = session.query(MD_NPT).filter_by(status=Status.PREPARED).limit(n_submit)
            self.submit_jobs(jobs_to_submit)

    def analyze(self, args: MonitorArgs):
        print('Analyzing results of md_npt')
        jobs_to_analyze = self.get_jobs_to_analyze(MD_NPT, n_analyze=args.n_analyze)
        results = self.analyze_multiprocess(self.analyze_single_job, jobs_to_analyze, args.n_jobs)
        for i, job in enumerate(jobs_to_analyze):
            result = results[i]
            job.result = json.dumps(result)
            if result.get('failed'):
                job.status = Status.FAILED
            elif result.get('continue'):
                job.status = Status.NOT_CONVERGED
            else:
                job.status = Status.ANALYZED
            session.commit()

    def analyze_single_job(self, job_dir: str,
                           check_converge: bool = True, cutoff_time: int = 7777):
        cwd = os.getcwd()
        os.chdir(job_dir)
        if isinstance(self.simulator, GROMACS):
            info_dict = dict()
            # TODO
            '''
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
                data = pv.data.GromacsParser(self.simulator.gmx_analysis).get_simulation_data(
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
                diffusion, _ = self.simulator.msd(gro='npt.xtc', tpr='npt.tpr', mol=True, begin=time_sim - 400)
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
                    expan, compr = self.simulator.get_fluct_props('npt.edr', begin=begin, end=end)
                    expan_list.append(expan)
                    compr_list.append(compr)
                expansion, expan_stderr = np.mean(expan_list), np.std(expan_list, ddof=1) / np.sqrt(nblock)
                compressi, compr_stderr = np.mean(compr_list), np.std(compr_list, ddof=1) / np.sqrt(nblock)
                expan_stderr = float('%.1e' % expan_stderr)  # 2 effective number for stderr
                compr_stderr = float('%.1e' % compr_stderr)  # 2 effective number for stderr

                temperature_and_stderr, pressure_and_stderr, potential_and_stderr, density_and_stderr, volume_and_stderr, \
                    ke_and_stderr, te_and_stderr, pv_and_stderr = \
                    self.gmx.get_properties_stderr(npt_edr,
                                                   ['Temperature', 'Pressure', 'Potential', 'Density', 'Volume',
                                                    'Kinetic-En.',
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
            '''
        else:
            raise ValueError

    def extend(self, args: MonitorArgs):
        dt = 0.002
        jobs_to_extend = session.query(MD_NPT).filter_by(status=Status.NOT_CONVERGED)
        if jobs_to_extend.count != 0:
            for job in jobs_to_extend:
                commands = []
                if isinstance(self.simulator, GROMACS):
                    gmx = self.simulator
                    os.chdir(job.ms_dir)
                    extend = json.loads(job.result)['continue_n'] * dt
                    commands += [gmx.convert_tpr(tpr='npt.tpr', extend=extend),
                                 gmx.mdrun(tpr='npt.tpr', ntomp=args.ntasks, exe=False, extend=True),
                                 'export GMX_MAXCONSTRWARN=-1',
                                 gmx.mdrun(tpr='hvap.tpr', ntomp=args.ntasks, rerun='npt.xtc', exe=False)]
                job.commands_extend = json.dumps(commands)
                job.status = Status.EXTENDED
                session.commit()

        jobs_to_submit = session.query(MD_NPT).filter_by(status=Status.EXTENDED)
        self.submit_jobs(jobs_to_submit=jobs_to_submit, extend=True)

    def update_fail_tasks(self):
        update_fail_mols()