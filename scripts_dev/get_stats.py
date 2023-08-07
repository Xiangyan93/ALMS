#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
CWD = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.join(CWD, '..'))
from tap import Tap
from alms.database.models import *


class Args(Tap):
    task: Literal['qm_cv', 'md_npt']
    """The task of molecular simulation."""


def main(args: Args):
    mols = session.query(SingleMolecule)
    print('There are total %i molecules.' % mols.count())
    print('%i molecules have been selected through active learning.' % mols.filter_by(active=True).count())
    print('%i molecules have been rejected through active learning.' % mols.filter_by(inactive=True).count())
    print('%i molecules haven\'t been considered in active learning.' %
          mols.filter_by(active=False, inactive=False).count())

    if args.task == 'qm_cv':
        jobs = session.query(QM_CV)
        for mol in session.query(SingleMolecule).filter_by(active=True):
            if Status.ANALYZED not in mol.status_qm_cv and Status.FAILED in mol.status_qm_cv:
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

