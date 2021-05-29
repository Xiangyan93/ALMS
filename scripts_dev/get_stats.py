#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
CWD = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.join(CWD, '..'))
from tap import Tap
from aims.database.models import *


class Args(Tap):
    task: Literal['qm_cv', 'md_npt']
    """The task of molecular simulation."""


def main(args: Args):
    if args.task == 'qm_cv':
        jobs = session.query(QM_CV)
        for mol in session.query(Molecule):
            if len(mol.status_qm_cv) == 1 and mol.status_qm_cv[0] == Status.FAILED:
                print(f'{mol.id} failed.')
    elif args.task == 'md_npt':
        jobs = session.query(MD_NPT)
    else:
        return

    print('There are total %i jobs' % jobs.count())
    for i, status in enumerate(['FAILED', 'STARTED', 'BUILD', 'PREPARED', 'SUBMITED', 'DONE', 'ANALYZED',
                                'NOT_CONVERGED', 'EXTENDED']):
        print('There are %i jobs in status %s.' % (jobs.filter_by(status=i-1).count(), status))


if __name__ == '__main__':
    main(args=Args().parse_args())

