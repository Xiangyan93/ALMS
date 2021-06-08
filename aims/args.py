#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
CWD = os.path.dirname(os.path.abspath(__file__))
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
from tap import Tap
import torch
import numpy as np
import pandas as pd
from .aimstools.jobmanager import Slurm


class SubmitArgs(Tap):
    smiles: List[str] = None
    """Submit a list of molecules in SMILES."""
    files: List[str] = None
    """Submit a list of SMILES in a file."""
    features_generator: List[str] = None
    """Method(s) of generating additional molfeatures."""
    heavy_atoms: Tuple[int, int] = None
    """Only molecules with heavy atoms in the interval will be submitted."""

    @property
    def smiles_list(self) -> List[str]:
        smiles = []
        if self.smiles is not None:
            smiles.extend(self.smiles)
        if self.files is not None:
            for file in self.files:
                df = pd.read_csv(file)
                for s in ['smiles', 'SMILES']:
                    if s in df:
                        smiles.extend(df[s].unique().tolist())
                else:
                  smiles.extend(df.iloc[:, 0].unique().tolist())
        return np.unique(smiles).tolist()


class KernelArgs(Tap):
    n_jobs: int = 1
    """The cpu numbers used for parallel computing."""


class SoftwareArgs(Tap):
    # QM softwares
    gaussian_exe: str = None
    """Executable file of GAUSSIAN"""
    # MD softwares
    packmol_exe: str = None
    """Executable file of packmol"""
    dff_root: str = None
    """Directory of Direct Force Field"""
    gmx_exe_analysis: str = None
    """"""
    gmx_exe_mdrun: str = None
    """"""


class MonitorArgs(SoftwareArgs):
    task: Literal['qm_cv', 'md_npt']
    """The task of molecular simulation"""
    # job manager args.
    job_manager: Literal['slurm'] = 'slurm'
    """Job manager of your cluster."""
    n_jobs: int = 8
    """The number of CPU cores used in the monitor."""
    partition: str
    """"""
    n_nodes: int = 1
    """The number of CPU nodes used in each slurm job."""
    n_cores: int = 8
    """The number of CPU cores used in each slurm job."""
    n_hypercores: int = None
    """The number of hyperthreading used in each slurm job."""
    n_gpu: int = 0
    """The number of GPU used in each slurm job."""
    mem: int = None
    """The memory used in each slurm job (MB)."""
    walltime: int = 48
    """Walltime of slurm jobs (hour)."""
    # controller args.
    n_prepare: int = 10
    """"""
    n_run: int = 20
    """The maximum number of running slurm jobs allowed."""
    n_analyze: int = 1000
    """"""
    t_sleep: int = 10
    """Sleep time for each iteration (mins). """
    # QM args
    conformer_generator: Literal['openbabel'] = 'openbabel'
    """The algorithm to generate 3D coordinates from SMILES."""
    n_conformer: int = 1
    """The number of conformers, this is only valid for QM calculations."""
    # GMX args
    n_gmx_multi: int = 1
    """The number of gmx jobs in each slurm job."""
    T_range: List[float] = [0.4, 0.9]
    """Reduced temperature range for simulations."""
    n_Temp: int = 8
    """Number of temperatures for simultions."""
    P_list: List[float] = [1]
    """Pressures for simulations."""
    stop_uncertainty: float = None
    """Tolerance of unsupervised active learning, should be a number from 0 to 1."""

    @property
    def job_manager_(self):
        return Slurm(partition=self.partition, n_nodes=self.n_nodes, n_cores=self.n_cores, n_gpu=self.n_gpu,
                     walltime=self.walltime)

    def process_args(self) -> None:
        ms_dir = os.path.join(CWD, '..', 'data', 'ms')
        if not os.path.exists(ms_dir):
            os.mkdir(ms_dir)

        if self.task == 'qm_cv':
            assert self.gaussian_exe is not None
            assert self.n_gpu == 0
        elif self.task == 'md_npt':
            assert self.packmol_exe is not None
            assert self.dff_root is not None
            assert self.gmx_exe_analysis is not None
            assert self.gmx_exe_mdrun is not None

        if self.n_gmx_multi == 1:
            assert self.n_gpu == 0

        if self.n_hypercores is None:
            self.n_hypercores = self.n_cores


class ExportArgs(Tap):
    property: Literal['density', 'cp'] = None
    """The property to export. None will output molecules list."""
