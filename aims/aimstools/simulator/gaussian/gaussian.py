#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import os
import numpy as np
from .. import BaseSimulator
from ...mol3D import Mol3D


class GaussianSimulator(BaseSimulator):
    def __init__(self, gauss_exe: str,
                 method: str = 'B3LYP', basis: str = '6-31G*', n_jobs: int = 1, memMB: int = None):
        self.gauss_exe = gauss_exe
        self.gauss_dir = os.path.dirname(gauss_exe)
        self.method = method
        self.basis = basis
        self.n_jobs = n_jobs
        self.memMB = memMB

    def prepare(self, smiles: str, path: str, name: str = 'gaussian', task: Literal['qm_cv'] = 'qm_cv',
                conformer_generator: Literal['openbabel'] = 'openbabel', seed: int = 0):
        mol3d = Mol3D(smiles, algorithm=conformer_generator, seed=seed)
        if task == 'qm_cv':
            self._create_gjf_cv(mol3d, path=path, name=name, T_list=np.arange(100, 900, 100))
        else:
            return

    def get_slurm_commands(self, file: str, tmp_dir: str):
        assert not os.path.exists(tmp_dir)
        commands = ['JOB_DIR=' + tmp_dir,
                    'mkdir -p ${JOB_DIR}',
                    'export GAUSS_EXEDIR=%s:%s/bsd' % (self.gauss_dir, self.gauss_dir),
                    'export GAUSS_SCRDIR=${JOB_DIR}', '%s %s' % (self.gauss_exe, file),
                    'rm -rf ${JOB_DIR}']
        return commands

    def analyze(self, log: str):
        pass

    def _create_gjf_cv(self, mol3d: Mol3D, path: str, name: str = 'gaussian',
                       scale: float = 0.9613, T_list: List[float] = None):
        gjf = os.path.join(path, '%s.gjf' % name)
        with open(gjf, 'w') as f:
            f.write('%%nprocshared=%d\n' % self.n_jobs)
            if self.memMB is not None:
                f.write('%%mem=%dMB\n' % self.memMB)
            f.write('%%chk=%(name)s.chk\n'
                    '# opt freq=hindrot %(method)s %(basis)s scale=%(scale).4f temperature=%(T).2f\n'
                    '\n'
                    'Title\n'
                    '\n'
                    '%(charge)i %(multiplicity)i\n'
                    % ({'name': name,
                        'method': self.method,
                        'basis': self.basis,
                        'scale': scale,
                        'T': T_list[0],
                        'charge': mol3d.charge,
                        'multiplicity': mol3d.spin
                        })
                    )
            for atom_line in mol3d.write(filetype='xyz').splitlines()[2:]:
                f.write(atom_line + '\n')
            f.write('\n')

            for T in T_list[1:]:
                f.write('--Link1--\n'
                        '%%chk=%(name)s.chk\n'
                        '# freq=(readfc,hindrot) geom=allcheck scale=%(scale).4f temperature=%(T).2f\n'
                        '\n'
                        % ({'name': name,
                            'scale': scale,
                            'T': T
                            })
                        )
