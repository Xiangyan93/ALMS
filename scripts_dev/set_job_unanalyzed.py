#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
CWD = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.join(CWD, '..'))
from tap import Tap
from alms.database.models import *


class Args(Tap):
    task: Literal['qm_cv', 'md_npt', 'md_binding']
    """The task of molecular simulation."""


def main(args: Args):
    if args.task == 'qm_cv':
        jobs = session.query(QM_CV)
    elif args.task == 'md_npt':
        jobs = session.query(MD_NPT)
    elif args.task == 'md_binding':
        jobs = session.query(MD_BINDING)
    else:
        return

    for job in jobs:
        if job.sh_file is not None:
            job.status = Status.SUBMITED
            job.result = None
    session.commit()


if __name__ == '__main__':
    main(args=Args().parse_args())

