#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from ..mol3D import Mol3D


class BaseSimulator():
    def set_system(self, smiles: str, save_dir: str):
        mol3d = Mol3D(smiles)
        mol3d.write(file=os.path.join(save_dir, 'mol.pdb'), filetype='pdb')
        mol3d.write(file=os.path.join(save_dir, 'mol.mol2'), filetype='mol2')
