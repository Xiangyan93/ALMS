#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import os
import shutil
from .base import GmxSimulation


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
        n_mol_list, pdb_list, mol2_list, length, box, vol = \
            super()._build(path, smiles_list, n_mol_list, n_mol_ratio, n_atoms, n_mols, length, density)
        print('Build coordinates using Packmol: %s molecules ...' % n_mol_list)
        self.packmol.build_box(pdb_list, n_mol_list, os.path.join(path, self.pdb), size=[i - 2 for i in box], silent=True)
        print('Create box using DFF ...')
        self.dff.build_box_after_packmol(mol2_list, n_mol_list, os.path.join(path, self.msd), mol_corr=os.path.join(path, self.pdb), size=box)

        # build msd for fast export
        self.packmol.build_box(pdb_list, [1] * len(pdb_list), os.path.join(path, self._single_pdb), size=box,
                               inp_file='build_single.inp', silent=True)
        self.dff.build_box_after_packmol(mol2_list, [1] * len(pdb_list), os.path.join(path, self._single_msd),
                                         mol_corr=os.path.join(path, self._single_pdb), size=box)

        if export:
            self.fast_export_single(ppf=ppf, gro_out='_single.gro', top_out='topol.top')
            self.gmx.pdb2gro(os.path.join(path, self.pdb), 'conf.gro', [i / 10 for i in box], silent=True)  # A to nm
            self.gmx.modify_top_mol_numbers('topol.top', n_mol_list)
            if ppf is not None:
                shutil.copy(os.path.join(ppf), 'ff.ppf')
