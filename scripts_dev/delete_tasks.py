#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tap import Tap
from alms.task import TASK
from alms.database.models import *


class Args(Tap):
    task: TASK
    """The task of molecular simulation."""


def main(args: Args):
    if args.task in ['qm_cv', 'md_npt', 'md_solvation']:
        tasks = session.query(SingleMoleculeTask)
    elif args.task == 'md_binding':
        tasks = session.query(DoubleMoleculeTask)
    else:
        return

    for task in tasks:
        task.delete()
    session.commit()


if __name__ == '__main__':
    main(args=Args().parse_args())

