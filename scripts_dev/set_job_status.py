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
    status: Literal[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    """Set all jobs to status."""


def main(args: Args):
    if args.task == 'qm_cv':
        jobs = session.query(QM_CV)
    elif args.task == 'md_npt':
        jobs = session.query(MD_NPT)
    else:
        return

    for job in jobs:
        job.status = args.status
    session.commit()


if __name__ == '__main__':
    main(args=Args().parse_args())

