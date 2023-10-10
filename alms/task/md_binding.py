#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tqdm import tqdm
import numpy as np
import pandas as pd
import re
from panedr.panedr import edr_to_df
from simutools.simulator.program import Packmol, PLUMED
from .base import BaseTask
from ..args import MonitorArgs
from ..database.models import *


em_mdp_kwargs = {'template': 't_em.mdp', 'mdp_out': 'em.mdp', 'dielectric': 1.0}


anneal_kwargs = {'template': 't_nvt_anneal.mdp', 'mdp_out': 'anneal.mdp', 'dt': 0.001,
                 'nsteps': 200000, 'nstenergy': 100, 'nstxout': 0, 'nstvout': 0, 'nstxtcout': 0, 'xtcgrps': 'System',
                 'dielectric': 1.0, 'T': 298, 'TANNEAL': 800, 'continuation': 'no'}


eq_kwargs = {'template': 't_npt.mdp', 'mdp_out': 'eq.mdp', 'dielectric': 1.0,
             'integrator': 'sd', 'dt': 0.001, 'nsteps': 1000000, 'nstenergy': 1000,
             'nstxout': 0, 'nstvout': 0, 'nstxtcout': 0, 'xtcgrps': 'System',
             'coulombtype': 'PME', 'rcoulomb': 1.2, 'rvdw': 1.2,
             'tcoupl': 'no', 'T': 298,
             'pcoupl': 'berendsen', 'tau_p': 1, 'compressibility': '4.5e-5', 'P': 1,
             'genvel': 'yes', 'seed': 0, 'constraints': 'h-bonds', 'continuation': 'no'}


class TaskBINDING(BaseTask):
    def __init__(self, job_manager: Slurm, force_field: Union[AMBER], simulator: Union[GROMACS],
                 plumed: PLUMED, packmol: Packmol):
        super().__init__(job_manager=job_manager)
        self.ff = force_field
        self.simulator = simulator
        self.packmol = packmol
        self.plumed = plumed

    def initiation(self, args: MonitorArgs):
        self.create_double_molecule_tasks(rule=args.combination_rule, file=args.combination_file)

    def active_learning(self, args: MonitorArgs):
        super().active_learning(args)

    def create(self, args: MonitorArgs):
        tasks = session.query(DoubleMoleculeTask).filter(DoubleMoleculeTask.active == True)
        for task in tqdm(tasks, total=tasks.count()):
            fail_jobs = [job for job in task.md_binding if job.status == Status.FAILED]
            if len(fail_jobs) <= 5:
                task.create_jobs(task='md_binding', n_repeats=5 + len(fail_jobs), T_list=[298.], P_list=[1.])
        session.commit()

    def build(self, args: MonitorArgs, length: float = 5., n_water: int = 3000, upper_bound: float = 2.):
        cwd = os.getcwd()
        # pick args.n_prepare tasks.
        tasks = []
        for task in session.query(DoubleMoleculeTask).filter(DoubleMoleculeTask.active == True):
            if Status.STARTED in task.status('md_binding'):
                tasks.append(task)
            if len(tasks) == args.n_prepare:
                break
        # checkout force field parameters for the molecules.
        for task in tqdm(tasks, total=len(tasks)):
            task.molecule_1.checkout(force_field=self.ff, simulator=self.simulator)
            task.molecule_2.checkout(force_field=self.ff, simulator=self.simulator)
        # Job.status: STARTED -> BUILD
        for task in tqdm(tasks, total=len(tasks)):
            mol1 = task.molecule_1
            mol2 = task.molecule_2
            for job in task.md_binding:
                if job.status != Status.STARTED:
                    continue
                os.chdir(job.ms_dir)
                # checkout tip3p water
                self.ff.checkout(smiles_list=['O'], n_mol_list=[1], name_list=['tip3p'],
                                 res_name_list=['SOL'], simulator=self.simulator)
                # create simulation box using packmol
                if task.self_task:
                    pdb_files = [f'{mol1.ms_dir}/{mol1.resname}.pdb', 'tip3p.pdb']
                    n_mol_list = [2, n_water]
                    mol_names = [mol1.resname, 'SOL']
                    top_dirs = [f'{mol1.ms_dir}/checkout', 'checkout']
                else:
                    pdb_files = [f'{mol1.ms_dir}/{mol1.resname}.pdb', f'{mol2.ms_dir}/{mol2.resname}.pdb', 'tip3p.pdb']
                    n_mol_list = [1, 1, n_water]
                    mol_names = [mol1.resname, mol2.resname, 'SOL']
                    top_dirs = [f'{mol1.ms_dir}/checkout', f'{mol2.ms_dir}/checkout', 'checkout']
                charges = mol1.formal_charge + mol2.formal_charge
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
                                           output='initial.pdb', box_size=[length] * 3, seed=job.seed)
                self.simulator.merge_top(top_dirs)
                for i, nmol in enumerate(n_mol_list):
                    self.simulator.modify_top_mol_numbers(top='topol.top', outtop='topol.top',
                                                          mol_name=mol_names[i], n_mol=nmol)
                self.simulator.convert_pdb(pdb='initial.pdb', tag_out='initial', box_size=[length] * 3)
                job.status = Status.BUILD
            session.commit()
        # Job.status: BUILD -> PREPARED
        # # prepares files for the jobs. All commands are saved and then submitted to SLURM.
        jobs = session.query(MD_BINDING).filter_by(status=Status.BUILD)
        for job in tqdm(jobs, total=jobs.count()):
            commands = []
            if isinstance(self.simulator, GROMACS):
                os.chdir(job.ms_dir)
                gmx = self.simulator
                # PLUMED
                natoms1 = job.double_molecule_task.molecule_1.GetNumAtoms
                natoms2 = job.double_molecule_task.molecule_2.GetNumAtoms
                self.plumed.generate_dat_from_template(template='bimolecule.dat', output='plumed.dat',
                                                       group1=f'1-{natoms1}',
                                                       group2=f'{natoms1 + 1}-{natoms1 + natoms2}',
                                                       upper_bound=upper_bound,
                                                       barrier=50)
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
                temp_kwargs['seed'] = job.seed
                gmx.generate_mdp_from_template(**temp_kwargs)
                commands += [gmx.grompp(mdp='eq.mdp', gro='anneal.gro', top='topol.top', tpr='eq.tpr', maxwarn=1,
                                        exe=False),
                             gmx.mdrun(tpr='eq.tpr', ntomp=args.ntasks, exe=False)]
                # NPT production with Langevin thermostat and Parrinello-Rahman barostat
                gmx.generate_mdp_from_template(
                    template='t_npt.mdp', mdp_out='npt.mdp', dielectric=1.0,
                    integrator='sd', dt=0.002, nsteps=15000000, nstenergy=50000,
                    nstxout=0, nstvout=0, nstxtcout=500000, xtcgrps='System',
                    coulombtype='PME', rcoulomb=1.2, rvdw=1.2,
                    tcoupl='no', T=job.T,
                    pcoupl='parrinello-rahman', tau_p=5, compressibility='4.5e-5', P=job.P,
                    genvel='no', seed=job.seed, constraints='h-bonds', continuation='yes')
                commands += [gmx.grompp(mdp='npt.mdp', gro='eq.gro', top='topol.top', tpr='npt.tpr', exe=False),
                             gmx.mdrun(tpr='npt.tpr', ntomp=args.ntasks, plumed='plumed.dat', exe=False)]
            else:
                raise ValueError
            job.commands_mdrun = json.dumps(commands)
            job.status = Status.PREPARED
            session.commit()
        os.chdir(cwd)

    def run(self, args: MonitorArgs):
        n_submit = args.n_run - self.job_manager.n_current_jobs
        if n_submit > 0:
            jobs_to_submit = session.query(MD_BINDING).filter_by(status=Status.PREPARED).limit(n_submit)
            self.submit_jobs(args=args, jobs_to_submit=jobs_to_submit)

    def analyze(self, args: MonitorArgs):
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

    def analyze_single_job(self, job_dir, check_converge: bool = True, cutoff_time: float = 60.):
        cwd = os.getcwd()
        os.chdir(job_dir)
        if isinstance(self.simulator, GROMACS):
            # general info
            info_dict = {}
            npt_edr = 'npt.edr'
            if not os.path.exists(npt_edr):
                os.chdir(cwd)
                return {'failed': True}
            df = edr_to_df(npt_edr)
            time_sim = df.Potential.index[-1] / 1000  # unit: ns
            info_dict['simulation_time'] = time_sim
            if time_sim < 25:
                os.chdir(cwd)
                return {'failed': True}
            elif time_sim < cutoff_time:
                info_dict['continue'] = True
                info_dict['continue_n'] = 5000000
                os.chdir(cwd)
                return info_dict
            # calculate binding free energy
            kernels = 'KERNELS'
            colvar = 'COLVAR'
            if not os.path.exists(kernels) or not os.path.exists(colvar):
                os.chdir(cwd)
                return {'failed': True}
            cv_min = 0.1
            cv_max = 2.0
            # get temperature
            pattern = r"T_([\d.]+)_P"
            matches = re.findall(pattern, job_dir)
            if matches:
                T = float(matches[-1])
            else:
                raise ValueError
            kbt = T * 0.0083144621  # kJ/mol
            time_column = 0
            cv_column = 1
            sigma_column = 2
            bias_column = 2
            # Get sigma
            df = pd.read_table(kernels, sep='\s+', dtype=float, comment='#', header=None)
            df = df[~df[time_column].duplicated(keep='last')]
            start_time = df[time_column].tolist()[0]
            stop_time = df[time_column].tolist()[-1] * 1.0
            sigma = df[df[time_column] == stop_time][sigma_column].tolist()[-1]
            # read COLVAR
            df = pd.read_table(colvar, sep='\s+', comment='#', header=None)
            df = df[~df[time_column].duplicated(keep='last')]
            df = df[(df[time_column] > start_time) & (df[time_column] <= stop_time)]
            # Reweight to get the free energy of OPES-MetaD
            nbins = 191
            FES_x = np.linspace(cv_min, cv_max, nbins)
            dist = (np.tile(df[cv_column].to_numpy(), (len(FES_x), 1)) - FES_x.reshape(-1, 1)) / sigma
            y = - 0.5 * dist * dist + df[bias_column].to_numpy().reshape(1, -1) / kbt
            FES_y = -kbt * np.logaddexp.reduce(y, axis=1)
            FES_y -= FES_y.min()
            # consider radius entropy effect
            FES_y = FES_y + kbt * np.log(4 * np.pi * FES_x ** 2)
            FES_y -= FES_y.min()
            info_dict_ = {'distance': FES_x.tolist(), 'fe': FES_y.tolist()}
            df_fes = pd.DataFrame(info_dict_)
            df_fes.to_csv('fes.dat', sep='\t', index=False, header=False)
            info_dict['binding_free_energy'] = FES_y[:int(nbins / 2)].min() - FES_y[-1]
            os.chdir(cwd)
            if info_dict['binding_free_energy'] < - 100:
                return {'failed': True}
            else:
                return info_dict
        else:
            raise ValueError

    def extend(self, args: MonitorArgs):
        n_extend = args.n_run - self.job_manager.n_current_jobs
        if n_extend > 0:
            dt = 0.002
            jobs_to_extend = session.query(MD_BINDING).filter_by(status=Status.NOT_CONVERGED).limit(n_extend)
            if jobs_to_extend.count != 0:
                for job in jobs_to_extend:
                    commands = []
                    if isinstance(self.simulator, GROMACS):
                        gmx = self.simulator
                        os.chdir(job.ms_dir)
                        extend = json.loads(job.result)['continue_n'] * dt
                        gmx.convert_tpr(tpr='npt.tpr', extend=extend)
                        commands += [gmx.mdrun(tpr='npt.tpr', ntomp=args.ntasks, plumed='plumed.dat', exe=False,
                                               extend=True)]
                    job.commands_extend = json.dumps(commands)
                    job.status = Status.EXTENDED
                    session.commit()

            jobs_to_submit = session.query(MD_BINDING).filter_by(status=Status.EXTENDED)
            self.submit_jobs(args=args, jobs_to_submit=jobs_to_submit, extend=True)

    def update_fail_tasks(self):
        tasks = session.query(DoubleMoleculeTask).filter(DoubleMoleculeTask.active == True)
        for task in tqdm(tasks, total=tasks.count()):
            fail_jobs = [job for job in task.md_binding if job.status == Status.FAILED]
            if len(fail_jobs) >= 5:
                task.active = False
                task.inactive = True
        session.commit()
