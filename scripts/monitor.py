#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from alms.database.models import *
from alms.args import MonitorArgs, SoftwareArgs
from alms.task.qm_cv import TaskCV
from alms.task.md_npt import TaskNPT
from alms.task.md_binding import TaskBINDING
from alms.task.md_solvation import TaskSOLVATION


def monitor(args: MonitorArgs):
    if args.task == 'qm_cv':
        task = TaskCV(simulator=args.Gaussian, job_manager=args.JobManager)
    elif args.task == 'md_npt':
        task = TaskNPT(force_field=args.ForceField, simulator=args.Simulator, packmol=args.Packmol,
                       job_manager=args.JobManager)
    elif args.task == 'md_binding':
        task = TaskBINDING(force_field=args.ForceField, simulator=args.Simulator, packmol=args.Packmol,
                           job_manager=args.JobManager, plumed=args.Plumed)
    elif args.task == 'md_solvation':
        task = TaskSOLVATION(force_field=args.ForceField, simulator=args.Simulator, packmol=args.Packmol,
                             job_manager=args.JobManager)
    else:
        raise ValueError()

    task.initiation(args)
    while True:
        print('Start a new loop\n'
              'Step1: active learning.\n\n')
        task.active_learning(args)
        print('Step2: create.\n\n')
        task.create(args)
        print('\nStep3: build.\n')
        task.build(args)
        print('\nStep4: run.\n')
        task.run(args)
        print('\nStep5: analyze.\n')
        task.analyze(args)
        print('\nStep6: extend.\n')
        task.extend(args)
        print('\nStep7: update failed mols.\n')
        task.update_fail_tasks(args)
        print('Sleep %d minutes...' % args.t_sleep)
        time.sleep(args.t_sleep * 60)


if __name__ == '__main__':
    monitor(args=MonitorArgs().parse_args())
