#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
from aims.args import MonitorArgs


def monitor(args: MonitorArgs):
    if args.task == 'qm_cv':
        from aims.qm.qm_cv import get_GaussianSimulator, create, build, run, analyze, extend
        simulator = get_GaussianSimulator(args)
    elif args.task == 'md_npt':
        from aims.md.md_npt import get_NptSimulator, create, build, run, analyze, extend
        simulator = get_NptSimulator(args)
    else:
        return
    job_manager = args.job_manager_

    create(args)
    while True:
        build(args, simulator)
        run(args, simulator, job_manager)
        analyze(args, simulator, job_manager)
        extend(args, simulator, job_manager)
        print('Sleep %d minutes...' % args.t_sleep)
        time.sleep(args.t_sleep * 60)


if __name__ == '__main__':
    monitor(args=MonitorArgs().parse_args())
