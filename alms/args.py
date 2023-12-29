#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
from tap import Tap
from functools import cached_property
from simutools.jobmanager import Slurm
from simutools.simulator.program import Gaussian, Packmol, GROMACS, PLUMED
from simutools.forcefields.amber import AMBER


class SubmitArgs(Tap):
    files: List[str]
    """Submit a list of input files in CSV format. It must contain smiles, optional for name."""
    heavy_atoms: Tuple[int, int] = None
    """Only molecules with heavy atoms in the interval will be submitted."""
    tag: str = None
    """tag for the submitted molecules."""
    excluded_smarts: List[str] = None


class TaskArgs(Tap):
    task: Literal['qm_cv', 'md_npt', 'md_binding', 'md_solvation']
    """The task of molecular simulation"""
    combination_rule: Literal['cross', 'full', 'self', 'specified'] = 'cross'
    """The combination rule to create double molecule tasks"""
    combination_file: str = None
    """The combination file to create double molecule tasks"""
    @property
    def task_nmol(self) -> int:
        if self.task in ['qm_cv', 'md_npt']:
            return 1
        else:
            return 2


class ALArgs(Tap):
    learning_type: Literal['all', 'explorative', 'exploitive', 'explorative_gpr_pu'] = 'all'
    """Active learning strategy"""
    model_config: str = None
    """The configuration file of the ML model"""
    batch_size: int = 1
    """number of samples added in each active learning iteration."""
    stop_cutoff: float = None
    """The cutoff of active learning"""
    n_query: int = None
    """number of samples to query in each active learning iteration. (default=None means query all samples in the 
    pool set)"""


class SoftwareArgs(Tap):
    # QM softwares
    gaussian: str = 'g16'
    """Executable file of GAUSSIAN"""
    # MD softwares
    packmol: str = 'packmol'
    """Executable file of packmol"""
    gmx: str = 'gmx'
    """Executable file of GROMACS"""
    force_field: str = 'antechamber'
    """The force field used for MD simulation."""
    plumed: str = 'plumed'
    """Executable file of PLUMED"""

    @cached_property
    def Gaussian(self) -> Gaussian:
        return Gaussian(exe=self.gaussian)

    @cached_property
    def Packmol(self) -> Packmol:
        return Packmol(exe=self.packmol)

    @cached_property
    def ForceField(self) -> Union[AMBER]:
        if self.force_field.endswith('antechamber'):
            return AMBER(exe=self.force_field, force_field='gaff2')
        else:
            raise ValueError(f'Unsupported force field: {self.force_field}')

    @cached_property
    def Simulator(self) -> Union[GROMACS]:
        if self.gmx is not None:
            return GROMACS(exe=self.gmx)
        else:
            raise ValueError(f'Unsupported MD simulation program: {self.gmx}')

    @cached_property
    def Plumed(self) -> PLUMED:
        return PLUMED(exe=self.plumed)


class JobManagerArgs(Tap):
    job_manager: Literal['slurm'] = 'slurm'
    """Job manager of your cluster."""
    partition: str
    """The partition for jobs."""
    n_nodes: int = 1
    """The number of CPU nodes used in each slurm job."""
    ntasks: int = 8
    """The number of CPU cores used in each slurm job."""
    n_gpu: int = 0
    """The number of GPU used in each slurm job."""
    mem: int = None
    """The memory used in each slurm job (GB)."""
    walltime: int = 48
    """Walltime of slurm jobs (hour)."""
    exclude: str = None
    """Do not submit job to the computational nodes."""

    @cached_property
    def JobManager(self) -> Slurm:
        try:
            job_manager = Slurm()
            job_manager.update_stored_jobs()
        except:
            job_manager = None
        return job_manager


class MonitorArgs(TaskArgs, ALArgs, SoftwareArgs, JobManagerArgs, Tap):
    n_jobs: int = 8
    """The number of CPU cores used in the monitor."""
    # controller args.
    n_prepare: int = 10
    """"""
    n_run: int = 20
    """The maximum number of running slurm jobs allowed."""
    n_analyze: int = 1000
    """The maximum number of analyzing jobs allowed."""
    n_task: int = 100
    """The maximum number of tasks allowed."""
    t_sleep: int = 10
    """Sleep time for each iteration (mins). """
    n_conformer: int = 1
    """The number of conformers, this is only valid for QM calculations."""
    T_range: List[float] = [0.4, 0.9]
    """Reduced temperature range for simulations."""
    n_Temp: int = 8
    """Number of temperatures for simultions."""
    P_list: List[float] = [1]
    """Pressures for simulations."""
    graph_kernel_type: Literal['graph', 'pre-computed'] = None
    """The type of kernel to use."""
    pool_size: int = None
    """
    A subset of the sample pool is randomly selected for active learning. 
    None means all samples are selected.
    """
    seed: int = 0
    """Random seed."""

    def process_args(self) -> None:
        if self.combination_file is not None:
            assert self.combination_rule == 'specified', ('combination_rule must be specified when combination_file is '
                                                          'not None.')
            assert self.task == 'md_binding', 'combination_file is only valid for md_binding task.'


class ExportArgs(Tap):
    property: Literal['density', 'cp', 'hvap', 'binding_free_energy'] = None
    """The property to export. None will output molecules list."""
