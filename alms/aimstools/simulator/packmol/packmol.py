#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import os
import random


class PackmolError(Exception):
    pass


class Packmol:
    def __init__(self, packmol_exe: str):
        self.packmol_exe = packmol_exe

    def build_box(self, pdb_files: List[str], n_mol_list: List[int], output: str,
                  box_size: Tuple[float, float, float],
                  slab: float = None, slab_multiple: bool = False,
                  tolerance: float = 1.8, seed: int = None,
                  inp_file='build.inp', silent: bool = False):
        assert len(pdb_files) > 0
        assert len(pdb_files) == len(n_mol_list)

        extensions = {filename.split('.')[-1].lower() for filename in pdb_files}
        if len(extensions) > 1:
            raise PackmolError('All file types should be the same')
        filetype = extensions.pop()

        seed = seed or random.randint(1e7, 1e8)

        inp = (
            'filetype {filetype}\n'
            'tolerance {tolerance}\n'
            'output {output}\n'
            'seed {seed}\n'.format(filetype=filetype, tolerance=tolerance, output=output, seed=seed)
        )

        # liquid-gas interface
        if slab is not None:
            box_liq = '0 0 0 %f %f %f' % (box_size[0], box_size[1], slab)
            box_gas = '0 0 %f %f %f %f' % (slab, box_size[0], box_size[1], box_size[2])
            for i, filename in enumerate(pdb_files):
                # put 1/50 molecules in gas phase. Do not put too many in case of nucleation in gas phase
                n_gas = n_mol_list[i] // 50
                n_liq = n_mol_list[i] - n_gas
                inp += (
                    'structure {filename}\n'
                    'number {n_liq}\n'
                    'inside box {box_liq}\n'
                    'end structure\n'
                    'structure {filename}\n'
                    'number {n_gas}\n'
                    'inside box {box_gas}\n'
                    'end structure\n'.format(filename=filename, n_liq=n_liq, n_gas=n_gas,
                                             box_liq=box_liq, box_gas=box_gas)
                )

        else:
            for i, filename in enumerate(pdb_files):
                number = n_mol_list[i]
                # slab model for multiple components
                if slab_multiple:
                    lz_per_slab = box_size[3] / len(n_mol_list)
                    box = '0 0 %f %f %f %f' % (i * lz_per_slab, box_size[0], box_size[1], (i + 1) * lz_per_slab)
                else:
                    box = '0 0 0 %f %f %f' % tuple(box_size)

                inp += (
                    'structure {filename}\n'
                    'number {number}\n'
                    'inside box {box}\n'
                    'end structure\n'.format(filename=filename, number=number, box=box)
                )

        with open(inp_file, 'w') as f:
            f.write(inp)

        # TODO subprocess PIPE not work for Packmol new version, do not know why.
        if silent:
            if os.name == 'nt':
                os.system(self.packmol_exe + ' < %s > nul' % inp_file)
            else:
                os.system(self.packmol_exe + ' < %s > /dev/null' % inp_file)
        else:
            os.system(self.packmol_exe + ' < %s' % inp_file)

            # (stdout, stderr) = (PIPE, PIPE) if silent else (None, None)
            # sp = subprocess.Popen([self.PACKMOL_BIN], stdin=PIPE, stdout=stdout, stderr=stderr)
            # sp.communicate(input=inp.encode())
