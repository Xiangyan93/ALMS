#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import math
from tqdm import tqdm
import numpy as np
import pandas as pd
import re
from panedr.panedr import edr_to_df
from simutools.simulator.program import Packmol
from .base import BaseTask
from .md_binding import em_mdp_kwargs, anneal_kwargs, eq_kwargs
from ..args import MonitorArgs
from ..database.models import *


class TaskSOLVATION(BaseTask):
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
            task.create_jobs(task='md_solvation', T_list=[298.], P_list=[1.])
        session.commit()

    def build(self, args: MonitorArgs, length: float = 5., n_water: int = 3000, upper_bound: float = 2.):
        cwd = os.getcwd()
        # pick args.n_prepare tasks.
        tasks = []
        for task in session.query(SingleMoleculeTask).filter(SingleMoleculeTask.active == True):
            if Status.STARTED in task.status('md_solvation'):
                tasks.append(task)
            if len(tasks) == args.n_prepare:
                break
        # checkout force field parameters for the molecules.
        for task in tqdm(tasks, total=len(tasks)):
            task.molecule.checkout(force_field=self.ff, simulator=self.simulator)
        # Job.status: STARTED -> BUILD
        for task in tqdm(tasks, total=len(tasks)):
            for job in task.md_solvation:
                if job.status != Status.STARTED:
                    continue
                os.chdir(job.ms_dir)
                # checkout tip3p water
                self.ff.checkout(smiles_list=['O'], n_mol_list=[1], name_list=['tip3p'],
                                 res_name_list=['SOL'], simulator=self.simulator)
                # create simulation box using packmol
                pdb_files = [f'{task.molecule.ms_dir}/{task.molecule.resname}.pdb', 'tip3p.pdb']
                n_mol_list = [1, n_water]
                mol_names = [task.molecule.resname, 'SOL']
                top_dirs = [f'{task.molecule.ms_dir}/checkout', 'checkout']
                charges = task.molecule.formal_charge
                if charges > 0:
                    self.ff.checkout(smiles_list=['[Cl-]'], n_mol_list=[1], name_list=['chloride'],
                                     res_name_list=['CL '], simulator=self.simulator, outname='chloride')
                    pdb_files += ['chloride.pdb']
                    n_mol_list += [charges]
                    top_dirs += ['chloride']
                    mol_names += ['CL']
                elif charges < 0:
                    self.ff.checkout(smiles_list=['[Na+]'], n_mol_list=[1], name_list=['sodium'],
                                     res_name_list=['NA '], simulator=self.simulator, outname='sodium')
                    pdb_files += ['sodium.pdb']
                    n_mol_list += [-charges]
                    top_dirs += ['sodium']
                    mol_names += ['NA']

                self.packmol.build_uniform(pdb_files=pdb_files,
                                           n_mol_list=n_mol_list,
                                           output='initial.pdb', box_size=[length] * 3, seed=0)
                self.simulator.merge_top(top_dirs)
                for i, nmol in enumerate(n_mol_list):
                    self.simulator.modify_top_mol_numbers(top='topol.top', outtop='topol.top',
                                                          mol_name=mol_names[i], n_mol=nmol)
                self.simulator.convert_pdb(pdb='initial.pdb', tag_out='initial', box_size=[length] * 3)
                job.status = Status.BUILD
            session.commit()
        # Job.status: BUILD -> PREPARED
        # # prepares files for the jobs. All commands are saved and then submitted to SLURM.
        jobs = session.query(MD_SOLVATION).filter_by(status=Status.BUILD)
        for job in tqdm(jobs, total=jobs.count()):
            commands = []
            if isinstance(self.simulator, GROMACS):
                os.chdir(job.ms_dir)
                gmx = self.simulator
                # energy minimization
                gmx.generate_mdp_from_template(**em_mdp_kwargs)
                commands += [gmx.grompp(mdp='em.mdp', gro='initial.gro', top='topol.top', tpr='em.tpr',
                                        maxwarn=2, exe=False),
                             gmx.mdrun(tpr='em.tpr', ntomp=args.ntasks, exe=False)]
                # NVT annealing from 0 to T_annealing to target T with Langevin thermostat
                temp_kwargs = anneal_kwargs.copy()
                temp_kwargs['T'] = job.T
                gmx.generate_mdp_from_template(**temp_kwargs)
                commands += [gmx.grompp(mdp='anneal.mdp', gro='em.gro', top='topol.top', tpr='anneal.tpr', exe=False),
                             gmx.mdrun(tpr='anneal.tpr', ntomp=args.ntasks, exe=False)]
                # NPT equilibrium with Langevin thermostat and Berendsen barostat
                temp_kwargs = eq_kwargs.copy()
                temp_kwargs['T'] = job.T
                temp_kwargs['P'] = job.P
                temp_kwargs['seed'] = 0
                gmx.generate_mdp_from_template(**temp_kwargs)
                commands += [gmx.grompp(mdp='eq.mdp', gro='anneal.gro', top='topol.top', tpr='eq.tpr', maxwarn=1,
                                        exe=False),
                             gmx.mdrun(tpr='eq.tpr', ntomp=args.ntasks, exe=False)]
                # Free energy perturbation
                for i in range(13):
                    os.mkdir(f'lambda{i}')
                    gmx.generate_mdp_from_template(
                        template='t_fep.mdp', mdp_out=f'fep_{i}.mdp', dielectric=1.0,
                        integrator='sd', dt=0.002, nsteps=1000000, nstenergy=10000,
                        nstxout=0, nstvout=0, nstxtcout=10000, xtcgrps='System',
                        coulombtype='PME', rcoulomb=1.2, rvdw=1.2,
                        tcoupl='no', T=job.T,
                        pcoupl='parrinello-rahman', tau_p=5, compressibility='4.5e-5', P=job.P,
                        genvel='no', seed=0, constraints='h-bonds', continuation='yes',
                        couple_moltype=job.single_molecule_task.molecule.resname,
                        init_lambda_state=i, fep_lambdas='0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.85 0.9 0.95 1.0')
                    commands += [f'cd lambda{i}',
                                 gmx.grompp(mdp=f'../fep_{i}.mdp', gro='../eq.gro', top='../topol.top',
                                            tpr=f'fep_{i}.tpr', exe=False),
                                 gmx.mdrun(tpr=f'fep_{i}.tpr', ntomp=args.ntasks, exe=False), 'cd ..']
            else:
                raise ValueError
            job.commands_mdrun = json.dumps(commands)
            job.status = Status.PREPARED
            session.commit()
        os.chdir(cwd)

    def run(self, args: MonitorArgs):
        n_submit = args.n_run - self.job_manager.n_current_jobs
        if n_submit > 0:
            jobs_to_submit = session.query(MD_SOLVATION).filter_by(status=Status.PREPARED).limit(n_submit)
            self.submit_jobs(args=args, jobs_to_submit=jobs_to_submit)

    def analyze(self, args: MonitorArgs):
        print('Analyzing results of md_binding')
        jobs_to_analyze = self.get_jobs_to_analyze(MD_BINDING, n_analyze=args.n_analyze)
        if len(jobs_to_analyze) == 0:
            return
        jobs_dirs = [job.ms_dir for job in jobs_to_analyze]
        results = self.analyze_multiprocess(self.analyze_single_job, jobs_dirs, args.n_jobs)
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

    def analyze_single_job(self, job_dir, check_converge: bool = True, cutoff_time: float = 100.):
        cwd = os.getcwd()
        os.chdir(job_dir)
        if isinstance(self.simulator, GROMACS):
            # general info
            info_dict = {}
            return info_dict
        else:
            raise ValueError

    def extend(self, args: MonitorArgs):
        pass

    def update_fail_tasks(self):
        pass
