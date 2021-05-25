#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import os
import time
import datetime
import subprocess
from subprocess import Popen, PIPE


class SlurmJob:
    def __init__(self, id: int, name: str, state: Literal['pending', 'running', 'done'],
                 work_dir: str = None, user: str = None, partition: str = None):
        self.id = id
        self.name = name
        self.state = state
        self.work_dir = work_dir
        self.user = user
        self.partition = partition

    def __repr__(self):
        return '<PbsJob: %i %s %s %s>' % (self.id, self.name, self.state, self.user)

    def __eq__(self, other):
        return self.id == other.id


class Slurm:
    def __init__(self, partition: str, n_nodes: int = 1, n_cores: int = 1, n_gpu: int = 0, walltime: int = 24,
                 env_cmd: bool = None):
        self.partition = partition
        self.n_nodes = n_nodes
        self.n_cores = n_cores
        self.n_gpu = n_gpu
        self.env_cmd = env_cmd or ''

        if os.name == 'nt':
            import win32api
            self.username = win32api.GetUserName()
        else:
            import pwd
            self.username = pwd.getpwuid(os.getuid()).pw_name

        self.stored_jobs = []
        self.walltime = walltime
        self.sh = '_job_slurm.sh'
        self.submit_cmd = 'sbatch'
        self.update_time = datetime.datetime.now()
        self.update_stored_jobs()

    @property
    def is_working(self) -> bool:
        cmd = 'sinfo --version'
        sp = Popen(cmd.split(), stdout=PIPE, stderr=PIPE)
        stdout, stderr = sp.communicate()
        return stdout.decode().startswith('slurm')

    @property
    def current_jobs(self) -> List[SlurmJob]:
        if (datetime.datetime.now() - self.last_update).total_seconds() >= 60:
            self.update_stored_jobs()
        return self.stored_jobs

    @property
    def n_current_jobs(self) -> int:
        return len(self.current_jobs)

    def generate_sh(self, path: str, name: str, commands: List[str], qos: str = None, n_gpu: int = None,
                    sh_index: bool = False) -> str:
        """

        Parameters
        ----------
        path: The directory to save the slurm.sh file.
        name: Slurm job name.
        commands: The commands to run.
        qos: Use qos priority.
        sh_index: Add index to the name.

        Returns
        -------
        The slurm file.
        """
        n_mpi, srun_commands = self._replace_mpirun_srun(commands)

        if n_gpu is not None:
            gpu_cmd = '#SBATCH --gres=gpu:%i\n' % n_gpu
        elif self.n_gpu is not None:
            gpu_cmd = '#SBATCH --gres=gpu:%i\n' % self.n_gpu
        else:
            gpu_cmd = ''

        if qos is not None:
            qos_cmd = '#SBATCH --qos=%s\n' % qos
        else:
            qos_cmd = ''

        if sh_index:
            name = name + '-%i' % self._get_local_index_of_job(path=path, name=name)
        file = os.path.join(path, name + '.sh')

        with open(file, 'w') as f:
            f.write('#!/bin/bash\n'
                    '#SBATCH -D %(workdir)s\n'
                    '#SBATCH -J %(name)s\n'
                    '#SBATCH -o %(out)s\n'
                    '#SBATCH -e %(err)s\n'
                    '#SBATCH -p %(queue)s\n'
                    '#SBATCH --time=%(time)i:00:00\n'
                    '#SBATCH --nodes=%(n_node)i\n'
                    '#SBATCH --ntasks=%(n_tasks)i\n'
                    '%(gpu_cmd)s'
                    '%(qos_cmd)s'
                    '\n'
                    '%(env_cmd)s\n\n'
                    % ({'name'   : name,
                        'out'    : name + '.out',
                        'err'    : name + '.err',
                        'queue'  : self.partition,
                        'time'   : self.walltime,
                        'n_node' : self.n_nodes,
                        'n_tasks': self.n_cores,
                        'gpu_cmd': gpu_cmd,
                        'qos_cmd': qos_cmd,
                        'env_cmd': self.env_cmd,
                        'workdir': path
                        })
                    )
            for cmd in srun_commands:
                f.write(cmd + '\n')
        return file

    def submit(self, file: str) -> bool:
        cmd = self.submit_cmd + ' ' + file
        return subprocess.call(cmd.split()) == 0

    def is_running(self, name) -> bool:
        job = self._get_job_from_name(name)
        if job is None:
            return False
        return job.state in ['pending', 'running']

    def kill_job(self, name: str) -> bool:
        job = self._get_job_from_name(name)
        if job is None:
            return False

        cmd = f'scancel {job.id}'
        return subprocess.call(cmd.split()) == 0

    def update_stored_jobs(self):
        print('Update job information')
        self.stored_jobs = []
        jobs = []
        for i in [3, 2, 1]:
            # in case get_all_jobs() raise Exception
            try:
                jobs = self._get_all_jobs()
            except Exception as e:
                print(repr(e))
            if jobs != []:
                break
            # TODO in case get_all_jobs() failed, retry after 1s
            if i > 1:
                time.sleep(1)
        for job in reversed(jobs):  # reverse the job list, in case jobs with same name
            if job not in self.stored_jobs:
                self.stored_jobs.append(job)

        self.stored_jobs.reverse()  # reverse the job list
        self.last_update = datetime.datetime.now()

    def _get_all_jobs(self) -> List[SlurmJob]:
        """get all jobs of current user.

        Returns
        -------
        A list of SlurmJob.
        """
        cmd = 'scontrol show job'
        sp = Popen(cmd.split(), stdout=PIPE, stderr=PIPE)
        stdout, stderr = sp.communicate()
        if sp.returncode != 0:
            print(stderr.decode())
            return []

        jobs = []
        for job_str in stdout.decode().split('\n\n'):  # split jobs
            if job_str.startswith('JobId'):
                job = self._get_job_from_str(job_str)
                # Show all jobs. Then check the user
                if job.user == self.username and job.partition == self.partition:
                    jobs.append(job)
        return jobs

    def _get_job_from_str(self, job_str) -> SlurmJob:
        """create job object from raw information."""
        work_dir = None
        for line in job_str.split():  # split properties
            try:
                key, val = line.split('=')[0:2]
            except:
                continue
            if key == 'JobId':
                id = int(val)
            elif key == 'UserId':
                user = val.split('(')[0]  # UserId=username(uid)
            elif key == 'JobName' or key == 'Name':
                name = val
            elif key == 'Partition':
                partition = val
            elif key == 'JobState':
                state_str = val
                if val in ('PENDING', 'RESV_DEL_HOLD'):
                    state = 'pending'
                elif val in ('CONFIGURING', 'RUNNING', 'COMPLETING', 'STOPPED', 'SUSPENDED'):
                    state = 'running'
                else:
                    state = 'done'
            elif key == 'WorkDir':
                work_dir = val
        job = SlurmJob(id=id, name=name, state=state, work_dir=work_dir, user=user, partition=partition)
        return job

    def _get_job_from_name(self, name: str) -> Optional[SlurmJob]:
        """get job information from job name."""
        # if several job have same name, return the one with the largest id (most recent job)
        for job in sorted(self.current_jobs, key=lambda x: x.id, reverse=True):
            if job.name == name:
                return job
        else:
            return None

    def _replace_mpirun_srun(self, commands: List[str]) -> Tuple[int, List[str]]:
        n_mpi = 1
        cmds_replaced = []
        for cmd in commands:
            if cmd.startswith('mpirun'):
                n_mpi = int(cmd.split()[2])
                cmd_srun = 'srun -n %i ' % n_mpi + ' '.join(cmd.split()[3:])
                cmds_replaced.append(cmd_srun)
            else:
                cmds_replaced.append(cmd)
        return n_mpi, cmds_replaced

    @staticmethod
    def _get_local_index_of_job(path: str, name: str):
        """the index assure no slurm jobs overwrite the existed sh file."""
        i = 0
        while os.path.exists(os.path.join(path, '%s-%i.sh' % (name, i))):
            i += 1
        return i
