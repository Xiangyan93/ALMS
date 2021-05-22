#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
from tqdm import tqdm
import json
import numpy as np
from aims.args import SubmitArgs
from aims.database import *
from aims.aimstools.rdkit.smiles import *
from aims.ml.chemprop.features.features_generators import get_features
from aims.ml.chemprop.args import PredictArgs
from aims.ml.chemprop.train import make_predictions


def mol_filter(smiles: str,
               smarts_bad: Dict[str, str],
               heavy_atom: Tuple[int, int] = None):
    mol = Chem.MolFromSmiles(smiles)
    # Return None for heavy atoms out of range.
    if heavy_atom is not None:
        if not heavy_atom[0] <= mol.GetNumAtoms() <= heavy_atom[1]:
            return None
    # Return None for wrong smiles.
    if mol is None:
        print('Ignore invalid SMILES: %s.' % smiles)
        return None
    # return None for molecules contain bad smarts.
    for name, smarts in smarts_bad.items():
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts),
                                          useChirality=True,
                                          maxMatches=1)
        if len(matches) != 0:
            return None
    # return canonical smiles.
    return get_rdkit_smiles(smiles)


def submit(args: SubmitArgs):
    smiles = args.smiles_list
    # Filter smiles using smarts patterns.
    smarts_bad = {
        # Not covered by TEAM FF
        'radicalC': '[#6;v0,v1,v2,v3]',
        '*=*=*': '*=*=*',
        '*#*~*#*': '*#*~*#*',
        '[F,Cl,Br]~[!#6]': '[F,Cl,Br]~[!#6]',
        '*#*~[!#6]': '*#*~[!#6]',
        '[NX2,NX4]': '[NX2,NX4]',
        'O~N(~[!$([OX1])])~[!$([OX1])]': 'O~N(~[!$([OX1])])~[!$([OX1])]',
        'peroxide': 'O~O',
        'N~N': 'N~N',
        '[O,N]*[O,N;H1,H2]': '[O,N]*[O,N;H1,H2]',
        'C=C~[O,N;H1,H2]': 'C=C~[O,N;H1,H2]',
        'beta-dicarbonyl': 'O=C~*~C=O',
        'a=*': 'a=*',
        'o': 'o',
        '[n;r5]': '[n;r5]',
        'pyridine-N-oxide': '[nX3;r6]',
        'triazine(zole)': '[$(nnn),$(nnan),$(nanan)]',
        '[R3]': '[R3]',
        '[r3,r4;R2]': '[r3,r4;R2]',
        '[r3,r4;#6X3]': '[r3,r4;#6X3]',
        '[r3,r4]~[!#6]': '[r3,r4]~[!#6]',
        'nitrate': 'O[NX3](~[OX1])~[OX1]',
        'amide': 'O=C[NX3]',
        'acyl-halide': 'O=C[F,Cl,Br]',
        'polybenzene': 'c1ccc2c(c1)cccc2',
        # Covered by TEAM FF but the results are not good
        '[r5;#6X3]': '[r5;#6X3]',
        '[r5]~[!#6]': '[r5]~[!#6]',
        'cyclo-ester': '[C;R](=O)O',
        'C=C~[O,N;H0]': 'C=C~[O,N;H0]',
        'C=C-X': 'C=C[F,Cl,Br]',
        '[F,Cl,Br][#6][F,Cl,Br]': '[F,Cl,Br][#6][F,Cl,Br]',
        'alkyne': '[CX2]#[CX2]',
        'acid': 'C(=O)[OH]',
        'nitrile': '[NX1]#[CX2][C,c]',
        'nitro': '[C,c][NX3](~[OX1])~[OX1]',
        'N-aromatic': 'n',
        'halogen': '[F,Cl,Br]',
    }
    smiles_valid = []
    for s in smiles:
        can_s = mol_filter(s, smarts_bad)
        if can_s is not None:
            smiles_valid.append(can_s)

    # Create molecules in database.
    for s in tqdm(np.unique(smiles_valid), total=np.unique(smiles_valid).size):
        mol = Molecule(smiles=s,
                       features=json.dumps(get_features(s, args.features_generator)))
        add_or_query(mol, ['smiles'])
    session.commit()


def predict(target: str):
    mols = session.query(Molecule)
    smiles = []
    features = []
    for mol in mols:
        smiles.append([mol.smiles])
        features.append(json.loads(mol.features)['rdkit_2d_normalized'])
    features = np.asarray(features)
    args = PredictArgs()
    args.checkpoint_dir = 'ml-models/%s' % target
    args.no_features_scaling = True
    args.process_args()
    preds = make_predictions(args, smiles, features, save_prediction=False)
    for i, mol in enumerate(mols):
        mol.update_dict('property_ml', {target: preds[i][0]})
    session.commit()


if __name__ == '__main__':
    submit(args=SubmitArgs().parse_args())
    predict('tt')
    predict('tb')
    predict('tc')
