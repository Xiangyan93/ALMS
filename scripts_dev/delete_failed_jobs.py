#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tap import Tap
from alms.task import TASK
from alms.database.models import *


class Args(Tap):
    task: TASK
    """The task of molecular simulation."""


def main(args: Args):
    if args.task == 'qm_cv':
        jobs = session.query(QM_CV)
    elif args.task == 'md_npt':
        jobs = session.query(MD_NPT)
    elif args.task == 'md_binding':
        jobs = session.query(MD_BINDING)
    elif args.task == 'md_solvation':
        jobs = session.query(MD_SOLVATION)
    else:
        return

    for job in jobs.filter_by(status=Status.FAILED):
        job.delete()
    session.commit()


if __name__ == '__main__':
    main(args=Args().parse_args())

