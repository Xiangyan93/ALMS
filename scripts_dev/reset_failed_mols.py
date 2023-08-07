#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
CWD = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.join(CWD, '..'))
from tap import Tap
from tqdm import tqdm
from alms.database.models import *


class Args(Tap):
    task: Literal['md_npt'] = 'md_npt'
    """The task of molecular simulation."""


def main(args: Args):
    if args.task == 'md_npt':
        mols = session.query(SingleMolecule).filter_by(active=True)
    else:
        return

    for mol in tqdm(mols, total=mols.count()):
        if len(mol.status_md_npt) == 1 and mol.status_md_npt[0] == Status.ANALYZED:
            continue
        mol.reset_md_npt()
    session.commit()


if __name__ == '__main__':
    main(args=Args().parse_args())

