#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rdkit.Chem.AllChem as Chem


def get_rdkit_smiles(smiles):
    rdk_mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(rdk_mol)
