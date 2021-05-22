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
    file: str = None
    """Submit a list of SMILES in a file."""
    features_generator: List[str] = None
    """Method(s) of generating additional molfeatures."""
    heavy_atoms: Tuple[int, int] = None
    """Only molecules with heavy atoms in the interval will be submitted."""

    @property
    def smiles_list(self):
        if self.smiles is not None:
            return self.smiles
        else:
            df = pd.read_csv(self.file)
            for s in ['smiles', 'SMILES']:
                if s in df:
                    return df[s].unique()
            else:
                return df.iloc[:, 0].unique()


class ActiveLearningArgs(Tap):
    stop_uncertainty: float = 0.2
    """Tolerance of unsupervised active learning, should be a number from 0 to 1."""
    rerun: bool = False
    """Rerun active learning from scratch."""
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
    n_prepare: int = 100
    """"""
    n_run: int = 50
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


class DMPNNArgs(Tap):
    no_cuda: bool = False
    """Turn off cuda (i.e., use CPU instead of GPU)."""
    gpu: int = None
    """Which GPU to use."""
    # Model arguments
    atom_messages: bool = False
    """Centers messages on atoms instead of on bonds."""
    bias: bool = False
    """Whether to add bias to linear layers."""
    hidden_size: int = 300
    """Dimensionality of hidden layers in MPN."""
    depth: int = 3
    """Number of message passing steps."""
    dropout: float = 0.0
    """Dropout probability."""
    activation: Literal['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'] = 'ReLU'
    """Activation function."""
    undirected: bool = False
    """Undirected edges (always sum the two relevant bond vectors)."""
    aggregation: Literal['mean', 'sum', 'norm'] = 'mean'
    """Aggregation scheme for atomic vectors into molecular vectors"""
    aggregation_norm: int = 100
    """For norm aggregation, number by which to divide summed up atomic 
    features"""

    @property
    def device(self) -> torch.device:
        """The :code:`torch.device` on which to load and process data and models."""
        if not self.cuda:
            return torch.device('cpu')

        return torch.device('cuda', self.gpu)

    @device.setter
    def device(self, device: torch.device) -> None:
        self.cuda = device.type == 'cuda'
        self.gpu = device.index

    @property
    def cuda(self) -> bool:
        """Whether to use CUDA (i.e., GPUs) or not."""
        return not self.no_cuda and torch.cuda.is_available()

    @cuda.setter
    def cuda(self, cuda: bool) -> None:
        self.no_cuda = not cuda

    @property
    def dataset_type(self):
        return 'regression'

    @property
    def num_tasks(self):
        return 1

    @property
    def max_data_size(self):
        return None


class TrainArgs(DMPNNArgs):
    data_path: str = None
    """Path to data CSV file."""
    smiles_columns: str = None
    """Column containing SMILES strings."""
    target_columns: str = None
    """Column containing target values."""
