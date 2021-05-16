#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
from aims.args import MonitorArgs
from aims.database import *


def monitor(args: MonitorArgs):
    if args.task == 'qm_cv':
        from aims.qm.qm_cv import get_GaussianSimulator, get_JobManager, create, prepare, run, analyze
        simulator = get_GaussianSimulator(args)
        job_manager = get_JobManager(args)
    else:
        return

    create(args)
    while True:
        prepare(args, simulator)
        run(args, simulator, job_manager)
        analyze(args, simulator, job_manager)
        print('Sleep %d minutes...' % args.t_sleep)
        time.sleep(args.t_sleep * 60)


if __name__ == '__main__':
    monitor(args=MonitorArgs().parse_args())
