#!/usr/bin/env python
# -*- coding: utf-8 -*-
from aims.args import MonitorArgs
from aims.database import *


def monitor(args: MonitorArgs):
    if args.task == 'qm_cv':
        from aims.qm.qm_cv import get_GaussianSimulator, get_JobManager, create, prepare, run
        simulator = get_GaussianSimulator(args)
        job_manager = get_JobManager(args)
    else:
        return
    create(args)
    prepare(args, simulator)
    run(args, simulator, job_manager)

if __name__ == '__main__':
    monitor(args=MonitorArgs().parse_args())
