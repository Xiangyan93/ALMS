#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
CWD = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.join(CWD, '..'))
from tap import Tap
from alms.database.models import *


class Args(Tap):
    task: Literal['qm_cv', 'md_npt', 'md_binding','md_solvation']
    """The task of molecular simulation."""

    @property
    def task_type(self):
        if self.task == 'md_binding':
            return DoubleMoleculeTask
        else:
            return SingleMoleculeTask


def main(args: Args):
    tasks = session.query(args.task_type)
    print('There are total %i tasks.' % tasks.count())
    print('%i tasks have been selected through active learning.' % tasks.filter_by(active=True).count())
    print('%i tasks have been rejected through active learning.' % tasks.filter_by(inactive=True).count())
    print('%i tasks haven\'t been considered in active learning.' %
          tasks.filter_by(active=False, inactive=False).count())

    if args.task == 'qm_cv':
        jobs = session.query(QM_CV)
        for task in session.query(SingleMoleculeTask).filter_by(active=True):
            if Status.ANALYZED not in task.status('qm_cv') and Status.FAILED in task.status('qm_cv'):
                print(f'{task.id} failed.')
    elif args.task == 'md_npt':
        jobs = session.query(MD_NPT)
    elif args.task == 'md_binding':
        jobs = session.query(MD_BINDING)
    elif args.task == 'md_solvation':
        jobs = session.query(MD_SOLVATION)
    else:
        return

    print('There are total %i jobs' % jobs.count())
    for i, status in enumerate(['FAILED', 'STARTED', 'BUILD', 'PREPARED', 'SUBMITED', 'DONE', 'ANALYZED',
                                'NOT_CONVERGED', 'EXTENDED']):
        print('There are %i jobs in status %s.' % (jobs.filter_by(status=i-1).count(), status))


if __name__ == '__main__':
    main(args=Args().parse_args())

