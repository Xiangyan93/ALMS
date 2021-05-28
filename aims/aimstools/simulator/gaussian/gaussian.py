#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import os
import math
import numpy as np
from ...mol3D import Mol3D


class GaussianSimulator:
    def __init__(self, gauss_exe: str,
                 method: str = 'B3LYP', basis: str = '6-31G*', n_jobs: int = 1, memMB: int = None):
        self.gauss_exe = gauss_exe
        self.gauss_dir = os.path.dirname(gauss_exe)
        self.method = method
        self.basis = basis
        self.n_jobs = n_jobs
        self.memMB = memMB

    def prepare(self, smiles: str, path: str, name: str = 'gaussian', task: Literal['qm_cv'] = 'qm_cv',
              conformer_generator: Literal['openbabel'] = 'openbabel', tmp_dir: str = '.', seed: int = 0) -> List[str]:
        mol3d = Mol3D(smiles, algorithm=conformer_generator, seed=seed)
        print('Build GAUSSIAN input file.')
        if task == 'qm_cv':
            self._create_gjf_cv(mol3d, path=path, name=name, T_list=np.arange(100, 1400, 100))
            file = os.path.join(path, '%s.gjf' % name)
            return self.get_slurm_commands(file, tmp_dir)
        else:
            return []

    def get_slurm_commands(self, file: str, tmp_dir: str):
        assert not os.path.exists(tmp_dir)
        commands = ['JOB_DIR=' + tmp_dir,
                    'mkdir -p ${JOB_DIR}',
                    'export GAUSS_EXEDIR=%s:%s/bsd' % (self.gauss_dir, self.gauss_dir),
                    'export GAUSS_SCRDIR=${JOB_DIR}', '%s %s' % (self.gauss_exe, file),
                    'rm -rf ${JOB_DIR}']
        return commands

    @staticmethod
    def analyze(log: str) -> Optional[Dict]:
        content = open(log).read()
        if content.find('Normal termination') == -1:
            return None
        if content.find('Error termination') > -1:
            return None
        if content.find('imaginary frequencies') > -1:
            return None

        result = {'EE': None, 'EE+ZPE': None, 'T': [], 'scale': [], 'cv': [], 'cv_corrected': [], 'FE': []}
        f = open(log)
        while True:
            line = f.readline()
            if line == '':
                break

            if line.strip().startswith('- Thermochemistry -'):
                line = f.readline()
                line = f.readline()
                T = float(line.strip().split()[1])
                result['T'].append(T)
                line = f.readline()
                if line.strip().startswith('Thermochemistry will use frequencies scaled by'):
                    scale = float(line.strip().split()[-1][:-1])
                else:
                    scale = 1
                result['scale'].append(scale)
            elif line.strip().startswith('E (Thermal)             CV                S'):
                line = f.readline()
                line = f.readline()
                Cv = float(line.strip().split()[2]) * 4.184
                result['cv'].append(Cv)
                line = f.readline()
                if line.strip().startswith('Corrected for'):
                    line = f.readline()
                    Cv_corr = float(line.strip().split()[3]) * 4.184
                    # Cv_corr might by NaN, this is a bug of gaussian
                    if math.isnan(Cv_corr):
                        Cv_corr = Cv
                    result['cv_corrected'].append(Cv_corr)
                else:
                    result['cv_corrected'].append(Cv)
            elif line.strip().startswith('Sum of electronic and thermal Free Energies='):
                fe = float(line.strip().split()[7])
                result['FE'].append(fe)
            elif result['EE+ZPE'] is None and line.strip().startswith('Sum of electronic and zero-point Energies='):
                ee_zpe = float(line.strip().split()[6])
                result['EE+ZPE'] = ee_zpe
            elif line.strip().startswith(' SCF Done:'):
                ee = float(line.strip().split()[3])
                result['EE'] = ee
        return result

    def _create_gjf_cv(self, mol3d: Mol3D, path: str, name: str = 'gaussian',
                       scale: float = 0.9613, T_list: List[float] = None):
        gjf = os.path.join(path, '%s.gjf' % name)
        with open(gjf, 'w') as f:
            f.write('%%nprocshared=%d\n' % self.n_jobs)
            if self.memMB is not None:
                f.write('%%mem=%dMB\n' % self.memMB)
            f.write('%%chk=%(path)s/%(name)s.chk\n'
                    '# opt freq=hindrot pop=full %(method)s %(basis)s scale=%(scale).4f temperature=%(T).2f\n'
                    '\n'
                    'Title\n'
                    '\n'
                    '%(charge)i %(multiplicity)i\n'
                    % ({'path': path,
                        'name': name,
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
                        '%%chk=%(path)s/%(name)s.chk\n'
                        '# freq=(readfc,hindrot) geom=allcheck scale=%(scale).4f temperature=%(T).2f\n'
                        '\n'
                        % ({'path': path,
                            'name': name,
                            'scale': scale,
                            'T': T
                            })
                        )
