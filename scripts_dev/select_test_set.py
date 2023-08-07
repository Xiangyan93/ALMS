#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
CWD = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.join(CWD, '..'))
from tap import Tap
from tqdm import tqdm
from alms.database.models import *
from sqlalchemy.sql import func


class Args(Tap):
    number: int = 0
    """The number of molecules selected as test set."""
    rule: Literal['random'] = 'random'
    """The rule to select test set molecules."""


def main(args: Args):
    mols = session.query(SingleMolecule).filter_by(active=False).order_by(func.random()).limit(args.number)
    for mol in tqdm(mols, total=mols.count()):
        mol.testset = True

    session.commit()


if __name__ == '__main__':
    main(args=Args().parse_args())

