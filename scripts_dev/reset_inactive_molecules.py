#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
CWD = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.join(CWD, '..'))
from tqdm import tqdm
from alms.database.models import *


def main():
    mols = session.query(Molecule)
    for mol in tqdm(mols, total=mols.count()):
        if mol.inactive == True and mol.fail == False:
            mol.inactive = False
    session.commit()


if __name__ == '__main__':
    main()

