#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import random
from openbabel import openbabel
import openbabel.pybel as pybel
from .saved_mol2 import get_smiles_mol2_dict


class Mol3D:
    def __init__(self, smiles: str, read_saved: bool = True, algorithm: Literal['openbabel'] = 'openbabel',
                 n_conformer: int = 0, seed: int = 0):
        try:
            self.mol_py = pybel.readstring('smi', smiles)
        except:
            raise RuntimeError('Cannot create molecule from SMILES using openbabel.')

        # 3D coordinates generate.
        self.smiles = smiles
        self.read_saved = read_saved
        self.algorithm = algorithm
        self.seed = seed
        self.smiles2mol2 = get_smiles_mol2_dict()

        if algorithm == 'openbabel':
            self.mol = self._mol_openbabel(minimize=True)
            self.conformers = self._conformers_openbabel(n_select=n_conformer)
        else:
            raise RuntimeError(f'Unknown 3D coordinates generate algorithm {algorithm}')

    @property
    def charge(self):
        return self.mol_py.charge

    @property
    def spin(self):
        return self.mol_py.spin

    @property
    def n_atoms(self) -> int:
        return len(self.mol_py.atoms)

    @property
    def molwt(self):
        return self.mol_py.molwt

    @property
    def formula(self):
        return self.mol_py.formula

    def write(self, file: str = None, filetype: Literal['pdb', 'mol2', 'xyz'] = 'mol2', resname: str = None):
        if self.algorithm == 'openbabel':
            if self.conformers:
                mol = self.conformers[0]
            else:
                mol = self.mol

            if file is not None:
                mol.write(filetype, file, overwrite=True)
            else:
                return mol.write(filetype)

    def _conformers_openbabel(self, n_select: int = 10, n_try: int = 10) -> List[pybel.Molecule]:
        if n_select == 0:
            return []
        random.seed(self.seed)
        ff = openbabel.OBForceField.FindForceField('mmff94')
        if n_try is None or n_try < n_select:
            n_try = n_select

        x_list = []
        for atom in self.mol.atoms:
            for x in atom.coords:
                x_list.append(x)
        xmin, xmax = min(x_list), max(x_list)
        xspan = xmax - xmin

        conformers = []
        for i in range(n_try):
            conformer = self._mol_openbabel(minimize=False)

            for atom in conformer.atoms:
                obatom = atom.OBAtom
                random_coord = [(random.random() * xspan + xmin) * k for k in [2, 1, 0.5]]
                obatom.SetVector(*random_coord)

            conformer.localopt()
            ff.Setup(conformer.OBMol)
            conformer.OBMol.SetEnergy(ff.Energy())
            conformers.append(conformer)
        conformers.sort(key=lambda x: x.energy)
        return conformers[:n_select]

    def _mol_openbabel(self, minimize=False) -> pybel.Molecule:
        try:
            mol = pybel.readstring('smi', self.smiles)
        except:
            raise RuntimeError('Cannot create molecule from SMILES using openbabel.')

        if self.read_saved and self.smiles in self.smiles2mol2:
            mol = next(pybel.readfile('mol2', self.smiles2mol2[self.smiles]))
        else:
            mol.addh()
            mol.make3D()
            if minimize:
                mol.localopt()
        return mol
