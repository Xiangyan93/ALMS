#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import os
import math
from .packmol import Packmol
from .dff import DFF
from ..mol3D import Mol3D
from ..utils import estimate_density_from_formula


class BaseSimulator:
    def __init__(self, packmol_exe: str, dff_root: str, dff_db: str = 'TEAMFF', dff_table: str = 'MGI'):
        self.packmol = Packmol(packmol_exe=packmol_exe)
        self.dff = DFF(dff_root=dff_root, default_db=dff_db, default_table=dff_table)
        self.n_atom_default: int = 3000
        self.n_mol_default: int = 75

        self.msd = 'init.msd'
        self.pdb = 'init.pdb'
        self._single_msd = '_single.msd'
        self._single_pdb = '_single.pdb'

    def _build(self, path: str, smiles_list: List[str],
               n_mol_list: List[int] = None, n_mol_ratio: List[int] = None,
               n_atoms: int = 3000, n_mols: int = 60, length: float = None, density: float = None,
               name_list: List[str] = None):
        pdb_list = []
        mol2_list = []
        n_components = len(smiles_list)
        n_atom_list = []  # number of atoms of each molecule
        molwt_list = []  # molecule weight of each molecule
        density_list = []  # estimated density of each molecule
        for i, smiles in enumerate(smiles_list):
            pdb = os.path.join(path, 'mol-%i.pdb' % i)
            mol2 = os.path.join(path, 'mol-%i.mol2' % i)
            mol3d = Mol3D(smiles)
            mol3d.write(pdb, filetype='pdb')
            mol3d.write(mol2, filetype='mol2')
            pdb_list.append(pdb)
            mol2_list.append(mol2)
            n_atom_list.append(mol3d.n_atoms)
            molwt_list.append(mol3d.molwt)
            density_list.append(estimate_density_from_formula(mol3d.formula) * 0.9)  # * 0.9, build box will be faster

        if n_mol_list is not None:
            n_mol_list = n_mol_list
        else:
            if n_mol_ratio is None:
                n_mol_ratio = [1] * n_components
            n_atom_all = sum([n_atom_list[i] * n for i, n in enumerate(n_mol_ratio)])
            n_atoms = n_atoms if n_atoms is not None else self.n_atom_default
            n_mols = n_mols if n_mols is not None else self.n_mol_default
            n_atoms_from_mol = n_atom_all * math.ceil(n_mols / sum(n_mol_ratio))
            n_atoms = max(n_atoms_from_mol, n_atoms)
            n_mol_list = [math.ceil(n_atoms / n_atom_all) * n for n in n_mol_ratio]

        mass = sum([molwt_list[i] * n_mol_list[i] for i in range(n_components)])

        if length is not None:
            length = length
            box_size = [length, length, length]
            vol = length ** 3
        else:
            if density is None:
                density = sum([density_list[i] * n_mol_list[i] for i in range(n_components)]) / \
                          sum(n_mol_list)
            vol = 10 / 6.022 * mass / density
            length = vol ** (1 / 3)  # assume cubic box
            box_size = [length, length, length]
        return n_mol_list, pdb_list, mol2_list, length, box_size, vol
