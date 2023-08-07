#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import json
import shutil
from simutools.jobmanager import Slurm


def update_dict(obj, attr: str, p_dict: Dict):
    content = getattr(obj, attr)
    if content is None:
        setattr(obj, attr, json.dumps(p_dict))
    else:
        d = json.loads(content)
        d.update(p_dict)
        setattr(obj, attr, json.dumps(d))


def update_list(obj, attr: str, p_list: List):
    content = getattr(obj, attr)
    if content is None:
        setattr(obj, attr, json.dumps(p_list))
    else:
        d = json.loads(content)
        d.extend(p_list)
        setattr(obj, attr, json.dumps(d))


def delete_job(job, session, job_manager: Slurm = None):
    # kill the slurm job if it is still running
    if job_manager is not None:
        if job_manager.is_running(job.slurm_name):
            job_manager.kill_job(job.slurm_name)
    # delete all files
    try:
        shutil.rmtree(job.ms_dir)
    except:
        pass

    session.delete(job)
    session.commit()
