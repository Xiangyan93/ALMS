#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
CWD = os.path.dirname(os.path.abspath(__file__))
import shutil
import subprocess
from subprocess import Popen, PIPE
from ...utils import random_string


class GmxError(Exception):
    pass


class GMX:
    TEMPLATE_DIR = os.path.join(CWD, 'template')

    def __init__(self, gmx_exe_analysis: str, gmx_exe_mdrun: str):
        self.GMX_EXE = gmx_exe_analysis
        self.GMX_MDRUN = gmx_exe_mdrun + ' mdrun'
        self.check_version()
        # TODO temporary hack for dielectric constant in mdp
        self._DIELECTRIC = 1.0
        # TODO temporary hack for LJ96 function
        self._LJ96 = False

    def check_version(self):
        cmd = '%s --version' % self.GMX_EXE
        try:
            sp = Popen(cmd.split(), stdout=PIPE, stderr=PIPE)
        except:
            raise GmxError('gmx not valid')
        stdout = sp.communicate()[0]
        for line in stdout.decode().splitlines():
            if line.startswith('GROMACS version'):
                break
        else:
            raise GmxError('gmx not valid')

        self.version = line.strip().split()[-1]
        if not (self.version.startswith('2016') or self.version.startswith('2018') or self.version.startswith('2019')):
            raise GmxError('Supported GROMACS versions: 2016.x, 2018.x, 2019.x')

        self.majorversion = self.version.split('.')[0]

    def grompp(self, mdp='grompp.mdp', gro='conf.gro', top='topol.top', tpr_out='md.tpr',
               cpt=None, maxwarn=3, silent=False, get_cmd=False):
        cmd = '%s -quiet -nobackup grompp -f %s -c %s -p %s -o %s -maxwarn %i' % (
            self.GMX_EXE, mdp, gro, top, tpr_out, maxwarn)
        if cpt is not None:
            cmd = '%s -t %s' % (cmd, cpt)
        if get_cmd:
            return cmd
        else:
            (stdout, stderr) = (PIPE, PIPE) if silent else (None, None)
            sp = Popen(cmd.split(), stdout=stdout, stderr=stderr)
            sp.communicate()

    def mdrun(self, name='md', nprocs=1, n_omp=None, rerun: str = None, extend=False, silent=False, get_cmd=False):
        # TODO temporary hack for LJ96 function
        if self._LJ96:
            n_omp = 1
        ###

        if n_omp is None:  # n_omp is None means auto
            for i in [6, 4, 2]:  # available OpenMP threads: 6, 4, 2
                if nprocs % i == 0:
                    n_omp = i
                    break
            else:
                n_omp = nprocs
        n_mpi = nprocs // n_omp

        # TODO temporary hack for LJ96 function
        if self._LJ96:
            if name.find('hvap') != -1:
                n_mpi = 1
        ###

        cmd = '%s -quiet -nobackup -ntomp %i -deffnm %s' % (self.GMX_MDRUN, n_omp, name)
        # always use mpirun even if only one process
        cmd = 'mpirun -np %i ' % n_mpi + cmd

        if rerun is not None:
            cmd = '%s -rerun %s' % (cmd, rerun)

        if extend:
            cmd = '%s -cpi %s' % (cmd, name + '.cpt')

        if get_cmd:
            return cmd
        else:
            (stdout, stderr) = (PIPE, PIPE) if silent else (None, None)
            sp = Popen(cmd.split(), stdout=stdout, stderr=stderr)
            sp.communicate()

    def minimize(self, gro, top, nprocs=1, silent=False, name='em', vacuum=False):
        if not vacuum:
            self.prepare_mdp_from_template('t_em.mdp')
        else:
            self.prepare_mdp_from_template('t_em_vacuum.mdp')

        self.grompp(gro=gro, top=top, tpr_out=name + '.tpr', silent=silent)
        self.mdrun(name=name, nprocs=nprocs, silent=silent)

    def dos(self, trr, tpr, T, group='System', log_out='dos.log', get_cmd=False, silent=False):
        cmd = '%s -quiet -nobackup dos -f %s -s %s -T %f -g %s' % (self.GMX_EXE, trr, tpr, T, log_out)
        if get_cmd:
            cmd = 'echo "%s" | %s' % (group, cmd)
            return cmd
        else:
            (stdout, stderr) = (PIPE, PIPE) if silent else (None, None)
            sp = Popen(cmd.split(), stdin=PIPE, stdout=stdout, stderr=stderr)
            sp.communicate(input=group.encode())

    def prepare_mdp_from_template(self, template, mdp_out='grompp.mdp', T=298, P=1, nsteps=10000, dt=0.001, TANNEAL=800,
                                  nstenergy=100, nstxout=0, nstvout=0, nstxtcout=10000, xtcgrps='System',
                                  restart=False, tcoupl='langevin', pcoupl='parrinello-rahman', gen_seed=-1,
                                  constraints='h-bonds', ppm=0, dielectric=None):
        template = os.path.join(GMX.TEMPLATE_DIR, template)
        if not os.path.exists(template):
            raise GmxError('mdp template not found')

        if tcoupl.lower() == 'langevin':
            integrator = 'sd'
            tcoupl = 'no'
            tau_t = str(0.001 / dt)  # inverse friction coefficient
        elif tcoupl.lower() == 'nose-hoover':
            integrator = 'md'
            tcoupl = 'nose-hoover'
            tau_t = '0.5'
        elif tcoupl.lower() == 'v-rescale':
            integrator = 'md'
            tcoupl = 'v-rescale'
            tau_t = '0.1'
        else:
            raise Exception('Invalid tcoupl, should be one of langvein, nose-hoover, v-rescale')

        if pcoupl.lower() == 'berendsen':
            tau_p = '1'
        elif pcoupl.lower() == 'parrinello-rahman':
            tau_p = '5'
        elif pcoupl.lower() == 'mttk':
            tau_p = '5'
            constraints = 'none'
        else:
            raise Exception('Invalid pcoupl, should be one of berendsen, parrinello-rahman, mttk')

        if restart:
            genvel = 'no'
            continuation = 'yes'
        else:
            genvel = 'yes'
            continuation = 'no'

        nstlist = max(1, int(0.01 / dt))

        if dielectric is None:
            dielectric = self._DIELECTRIC

        with open(template) as f_t:
            contents = f_t.read()
        contents = contents.replace('%T%', str(T)).replace('%P%', str(P)).replace('%nsteps%', str(int(nsteps))) \
            .replace('%dt%', str(dt)).replace('%nstenergy%', str(nstenergy)) \
            .replace('%nstxout%', str(nstxout)).replace('%nstvout%', str(nstvout)) \
            .replace('%nstxtcout%', str(nstxtcout)).replace('%xtcgrps%', str(xtcgrps)) \
            .replace('%genvel%', genvel).replace('%seed%', str(gen_seed)).replace('%continuation%', continuation) \
            .replace('%integrator%', integrator).replace('%tcoupl%', tcoupl).replace('%tau-t%', tau_t) \
            .replace('%pcoupl%', pcoupl).replace('%tau-p%', tau_p) \
            .replace('%constraints%', constraints).replace('%TANNEAL%', str(TANNEAL)).replace('%ppm%', str(ppm)) \
            .replace('%nstlist%', str(nstlist)).replace('%dielectric%', str(dielectric))

        with open(mdp_out, 'w') as f_mdp:
            f_mdp.write(contents)

    def energy(self, edr, properties: [str], begin=0, end=None, skip=None, fluct_props=False, get_cmd=False, out=None):
        cmd = '%s -quiet -nobackup energy -f %s -b %s' % (self.GMX_EXE, edr, str(begin))
        if end is not None:
            cmd += ' -e %s' % (str(end))
        if skip is not None:
            cmd += ' -skip %s' % (str(skip))
        if out is not None:
            cmd += ' -o %s' % (str(out))
        if fluct_props:
            cmd += ' -fluct_props'
        if get_cmd:
            property_str = '\\n'.join(properties)
            cmd = 'echo -e "%s" | %s' % (property_str, cmd)
            return cmd
        else:
            sp = Popen(cmd.split(), stdout=PIPE, stdin=PIPE, stderr=PIPE)
            property_str = '\n'.join(properties)
            out, err = sp.communicate(input=property_str.encode())
            return out

    def get_fluct_props(self, edr, begin=0, end=None) -> (float, float):
        '''
        Get thermal expansion and compressibility using fluctuation of enthalpy, volume
        Only works for NPT simulation
        :param edr:
        :param begin:
        :param end:
        :return:
        '''
        sp_out = self.energy(edr, properties=['temp', 'vol', 'enthalpy'], begin=begin, end=end, fluct_props=True)

        expansion = None
        compressibility = None
        for line in sp_out.decode().splitlines():
            if line.startswith('Coefficient of Thermal Expansion Alpha_P'):
                expansion = float(line.split()[-2])
            elif line.startswith('Isothermal Compressibility Kappa'):
                compressibility = float(line.split()[-2]) * 1e5
        return expansion, compressibility

    def get_properties_stderr(self, edr, properties: [str], begin=0, end=None) -> [[float]]:
        sp_out = self.energy(edr, properties=properties, begin=begin, end=end)

        lines = sp_out.decode().splitlines()
        results = []
        for prop in properties:
            for line in lines:
                if prop in ['Kinetic-En.', 'Total-Energy']:
                    if line.lower().startswith(prop.lower().replace('-', ' ')):
                        results.append([float(line.split()[2]), float(line.split()[3])])
                        break
                else:
                    if line.lower().startswith(prop.lower()):
                        results.append([float(line.split()[1]), float(line.split()[2])])
                        break
            else:
                raise GmxError('Invalid property')
        return results

    def get_property_stderr(self, edr, prop: str, begin=0, end=None) -> [[float]]:
        return self.get_properties_stderr(edr, [prop], begin, end)[0]

    def get_box(self, edr, begin=0) -> [float]:
        sp = subprocess.Popen([self.GMX_EXE, 'energy', '-f', edr, '-b', str(begin)], stdout=PIPE, stdin=PIPE,
                              stderr=PIPE)
        sp_out = sp.communicate(input='Box-'.encode())[0]

        box = [0, 0, 0]
        for line in sp_out.decode().splitlines():
            if line.startswith('Box-X'):
                box[0] = float(line.split()[1])
            if line.startswith('Box-Y'):
                box[1] = float(line.split()[1])
            if line.startswith('Box-Z'):
                box[2] = float(line.split()[1])
        return box

    def density(self, trj, tpr, xvg='density.xvg', group='System', begin=0, end=None,
                center=False, silent=False, get_cmd=False):
        cmd = '%s -quiet -nobackup density -f %s -s %s -b %f -o %s' % (self.GMX_EXE, trj, tpr, begin, xvg)
        if end is not None:
            cmd += ' -e %f' % end
        inp = group
        if center:
            cmd += ' -center'
            inp = '%s\n%s' % (group, group)

        if get_cmd:
            cmd = 'echo "%s" | %s' % (inp, cmd)
            return cmd
        else:
            (stdout, stderr) = (PIPE, PIPE) if silent else (None, None)
            sp = Popen(cmd.split(), stdin=PIPE, stdout=stdout, stderr=stderr)
            sp.communicate(input=inp.encode())

    @staticmethod
    def scale_box(gro, gro_out, new_box: [float]):
        '''
        Scale gro box to desired size.
        The coordinate of all atoms are scaled.
        The velocities are not modified.
        Only support rectangular box
        '''
        NAtoms = 0
        nresidue = []
        residue = []
        element = []
        natom = []
        xyz = []
        vxyz = []
        box = []
        if not os.path.exists(gro):
            raise GmxError('gro not found')
        with open(gro) as f_gro:
            lines = f_gro.read().splitlines()

        for i, line in enumerate(lines):
            n = i + 1
            if n == 1:
                title = line
            if n == 2:
                NAtoms = int(line.strip())
                continue
            if n > 2 and n <= 2 + NAtoms:
                nresidue.append(int(line[:5]))
                residue.append(line[5:10])
                element.append(line[10:15])
                natom.append(int(line[15:20]))
                x = float(line[20:28])
                y = float(line[28:36])
                z = float(line[36:44])
                vx = float(line[44:52])
                vy = float(line[52:60])
                vz = float(line[60:68])
                xyz.append([x, y, z])
                vxyz.append([vx, vy, vz])
            if n == 3 + NAtoms:
                box = [float(word) for word in line.strip().split()[:3]]
                break

        scale = [new_box[i] / box[i] for i in range(3)]
        xyz = [[i[0] * scale[0], i[1] * scale[1], i[2] * scale[2]] for i in xyz]

        with open(gro_out, 'w') as f_out:
            f_out.write('Scaled : %s\n%i\n' % (title, NAtoms))
            for i in range(NAtoms):
                f_out.write('%5i%5s%5s%5i%8.3f%8.3f%8.3f%8.4f%8.4f%8.4f\n'
                            % (nresidue[i], residue[i], element[i], natom[i],
                               xyz[i][0], xyz[i][1], xyz[i][2], vxyz[i][0], vxyz[i][1], vxyz[i][2]))
            f_out.write('%f %f %f\n' % (new_box[0], new_box[1], new_box[2]))

    @staticmethod
    def generate_top(itp, molecules, numbers):
        shutil.copy(itp, 'topol.top')
        with open('topol.top', 'a') as f:
            f.write('\n[system]\n[molecules]\n')
            for i, molecule in enumerate(molecules):
                f.write('%s %i\n' % (molecule, numbers[i]))

    @staticmethod
    def get_top_mol_numbers(top):
        with open(top) as f:
            lines = f.read().splitlines()

        newlines = []
        mols = []
        START = False
        for line in lines:
            if line.find('[') != -1 and line.find('molecules') != -1:  # [ molecules ]
                START = True
                newlines.append(line)
                continue
            if not START:
                newlines.append(line)
            if START and not line.strip() == '' and not line.startswith(';'):
                words = line.strip().split()
                mols.append([words[0], int(words[1])])
        return mols

    @staticmethod
    def modify_top_mol_numbers(top, numbers):
        with open(top) as f:
            lines = f.read().splitlines()

        newlines = []
        mols = []
        START = False
        for line in lines:
            if line.find('[') != -1 and line.find('molecules') != -1:  # [ molecules ]
                START = True
                newlines.append(line)
                continue
            if not START:
                newlines.append(line)
            if START and not line.strip() == '' and not line.startswith(';'):
                words = line.strip().split()
                mols.append([words[0], int(words[1])])
        if len(mols) != len(numbers):
            raise GmxError('Type of molecules in top not consistent')

        for i, [mol_name, _] in enumerate(mols):
            newlines.append('%s\t%i' % (mol_name, numbers[i]))

        with open(top, 'w')  as f:
            f.write('\n'.join(newlines))

    def replicate_gro(self, gro, top, nbox, silent=True):
        from functools import reduce
        import operator

        gro_tmp = random_string(8) + '.gro'
        cmd = '%s -quiet genconf -f %s -o %s -nbox %i %i %i' % (self.GMX_EXE, gro, gro_tmp, nbox[0], nbox[1], nbox[2])
        (stdout, stderr) = (PIPE, PIPE) if silent else (None, None)
        sp = Popen(cmd.split(), stdout=stdout, stderr=stderr)
        sp.communicate()
        shutil.move(gro_tmp, gro)

        with open(top) as f:
            content = f.read()

        LASTLINE = '\n'
        LAST = False
        for line in content.splitlines():
            if line.find('molecules') > -1 and line.find('[') > -1:
                LAST = True
                continue
            if LAST and not line.strip() == '' and not line.startswith(';'):
                LASTLINE += line + '\n'
        LASTLINE *= (reduce(operator.mul, nbox, 1) - 1)

        with open(top, 'a') as f:
            f.write(LASTLINE)

    def pdb2gro(self, pdb, gro_out, box: [float], silent=False):
        if len(box) != 3:
            raise GmxError('Invalid box')

        (stdout, stderr) = (PIPE, PIPE) if silent else (None, None)
        sp = Popen([self.GMX_EXE, 'editconf', '-f', pdb, '-o', gro_out, '-box', str(box[0]), str(box[1]), str(box[2])],
                   stdin=PIPE, stdout=stdout, stderr=stderr)
        sp.communicate()

    def velacc(self, trr, tpr=None, group=None, begin=0, xvg_out='velacc', silent=False):
        if tpr is None:
            tpr = trr
        if group is None:
            raise GmxError('No group specifed')

        (stdout, stderr) = (PIPE, PIPE) if silent else (None, None)
        sp = Popen([self.GMX_EXE, 'velacc', '-f', trr, '-s', tpr, '-o', xvg_out, '-b', str(begin), '-mol',
                    '-nonormalize'],
                   stdin=PIPE, stdout=stdout, stderr=stderr)
        sp.communicate(input=str(group).encode())

    def diffusion(self, xtc, tpr, group='System', mol=False, begin=0, end=None, xvg_out='msd.xvg',
                  beginfit=-1, endfit=-1):
        cmd = '%s -quiet -nobackup msd -f %s -s %s -o %s -b %s -beginfit %s -endfit %s' % (
            self.GMX_EXE, xtc, tpr, xvg_out, str(begin), str(beginfit), str(endfit))
        if end is not None:
            cmd += ' -e %s' % str(end)
        if mol:
            # calculate the MSD of COM of molecules
            cmd += ' -mol'

        sp = Popen(cmd.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
        out, err = sp.communicate(input=str(group).encode())

        for line in out.decode().splitlines():
            if line.startswith('D['):
                words = line.strip().split(']')[-1].strip().split()
                diffusion = float(words[0])
                stderr = float(words[2][:-1])
                unit = float(words[3])
                diffusion *= unit  # cm^2/s
                stderr *= unit  # cm^2/s
                return diffusion, stderr

        raise GmxError('Error running gmx msd')

    def msd_com(self, xtc, tpr, resname, beginfit=-1, endfit=-1, xvg_out=None, silent=False):
        ndx = 'com-' + resname + '.ndx'
        GMX.select_com(tpr, resname, ndx_out=ndx)
        if xvg_out is None:
            xvg_out = 'msd-com-%s.xvg' % resname

        (stdout, stderr) = (PIPE, PIPE) if silent else (None, None)
        sp = Popen([self.GMX_EXE, 'msd', '-f', xtc, '-s', tpr, '-n', ndx, '-o', xvg_out, '-nomw',
                    '-beginfit', str(beginfit), '-endfit', str(endfit)], stdout=stdout, stderr=stderr)
        sp_out = sp.communicate()[0]

        for line in sp_out.decode().splitlines():
            if line.startswith('D['):
                return line
        raise GmxError('Error running gmx msd')

    def traj_com(self, xtc, tpr, trj_out='com.xtc', begin=0, end=0, silent=False):
        ndx = 'com.ndx'
        self.select_com(tpr, 'all', ndx_out=ndx)

        (stdout, stderr) = (PIPE, PIPE) if silent else (None, None)
        cmd = '%s -quiet -nobackup traj -s %s -f %s -oxt %s -com -mol -n %s -b %f -e %f' % (
            self.GMX_EXE, tpr, xtc, trj_out, ndx, begin, end)
        sp = Popen(cmd.split(), stdout=stdout, stderr=stderr)
        sp.communicate()
        return trj_out, ndx

    def select_com(self, tpr, resname='all', ndx_out='com.ndx'):
        cmd = '%s -quiet -nobackup select -s %s -on %s' % (self.GMX_EXE, tpr, ndx_out)
        sp = Popen(cmd.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
        if resname == 'all':
            select_com_str = 'res_com of all'
        else:
            select_com_str = 'res_com of resname %s' % resname
        sp.communicate(input=select_com_str.encode())

    @staticmethod
    def generate_top_for_hvap(top, top_out):
        with open(top) as f:
            lines = f.read().splitlines()
        lines = [l for l in lines if not (l.startswith(';') or l == '')]

        f_out = open(top_out, 'w')

        line_number_molecule = []
        line_number_atom = []
        line_number_system = None
        for n, line in enumerate(lines):
            if line.find('[') != -1 and line.find('moleculetype') != -1:
                line_number_molecule.append(n)
            if line.find('[') != -1 and line.find('atoms') != -1:
                line_number_atom.append(n)
            if line.find('[') != -1 and line.find('system') != -1:
                line_number_system = n

        n_molecules = len(line_number_molecule)

        for n in range(line_number_molecule[0]):
            f_out.write(lines[n] + '\n')

        for i in range(n_molecules):
            for n in range(line_number_molecule[i], line_number_atom[i]):
                f_out.write(lines[n] + '\n')
            line_number_next_section = line_number_molecule[i + 1] if i < n_molecules - 1 else line_number_system
            if line_number_next_section is None:
                line_number_next_section = len(lines)
            n_atoms = 0
            f_out.write('[ atoms ]\n')
            for n in range(line_number_atom[i] + 1, line_number_next_section):
                line = lines[n]
                if line.find('[') != -1 or line.startswith('#'):
                    f_out.write('[ bonds ]\n[ exclusions ]\n')
                    for i in range(1, n_atoms):
                        exclusions = range(i, n_atoms + 1)
                        f_out.write(' '.join(list(map(str, exclusions))) + '\n')
                    break
                f_out.write(line + '\n')
                n_atoms += 1

        if line_number_system is not None:
            for n in range(line_number_system, len(lines)):
                f_out.write(lines[n] + '\n')

    def cut_traj(self, trr, trr_out, begin, end, silent=False):
        cmd = '%s -quiet -nobackup trjconv -f %s -o %s -b %s -e %s' % (
            self.GMX_EXE, trr, trr_out, str(begin), str(end))
        (stdout, stderr) = (PIPE, PIPE) if silent else (None, None)
        sp = Popen(cmd.split(), stdin=PIPE, stdout=stdout, stderr=stderr)
        sp.communicate()

    def slice_gro_from_traj(self, trr, tpr, gro_out, begin, end, dt, silent=False):
        cmd = '%s -quiet -nobackup trjconv -f %s -s %s -o %s -b %s -e %s -dt %s -sep -pbc whole' % (
            self.GMX_EXE, trr, tpr, gro_out, str(begin), str(end), str(dt))
        (stdout, stderr) = (PIPE, PIPE) if silent else (None, None)
        sp = Popen(cmd.split(), stdin=PIPE, stdout=stdout, stderr=stderr)
        sp.communicate(input='System'.encode())

    def extend_tpr(self, tpr, extend, silent=False):
        cmd = '%s -quiet convert-tpr -s %s -o %s -extend %s' % (self.GMX_EXE, tpr, tpr, str(extend))
        (stdout, stderr) = (PIPE, PIPE) if silent else (None, None)
        sp = Popen(cmd.split(), stdout=stdout, stderr=stderr)
        sp.communicate()

    def trjconv(self, tpr, input_trj, output_trj, pbc_nojump=False, skip=1, end=None, silent=False, select='System',
                get_cmd=False):
        cmd = '%s -quiet -nobackup trjconv -s %s -f %s -o %s -skip %i' % (
            self.GMX_EXE, tpr, input_trj, output_trj, skip)
        if end is not None:
            cmd += ' -e %i' % (end)
        if pbc_nojump:
            cmd += ' -pbc nojump'
        if get_cmd:
            return 'echo %s | ' % (select) + cmd
        else:
            (stdout, stderr) = (PIPE, PIPE) if silent else (None, None)
            sp = Popen(cmd.split(), stdin=PIPE, stdout=stdout, stderr=stderr)
            sp.communicate(input=select.encode())

    def generate_gpu_multidir_cmds(self, dirs: [str], commands: [str], n_parallel, n_gpu=0, n_omp=None,
                                   n_procs=None) -> [[str]]:
        '''
        Set n_omp in most case. If n_procs is set, n_omp has no effect.
        :param dirs:
        :param commands:
        :param n_parallel:
        :param n_gpu:
        :param n_procs:
        :param n_omp:
        :return:
        '''
        import math, re

        def replace_gpu_multidir_cmd(dirs: [str], cmd: str) -> str:
            n_multi = len(dirs)
            if cmd.startswith('export '):
                pass

            elif cmd.find('gmx') != -1 and cmd.find(' mdrun ') != -1:
                n_mpi = n_multi
                n_thread = n_omp

                if n_procs is not None:
                    if cmd.find('mpirun ') != -1 and cmd.find('mpirun -np 1 ') == -1:
                        # optimize n_mpi and n_thread
                        # GPU tasks prefer more n_mpi
                        # ensure n_mpi equals 2*n
                        # do not optimize for mpirun -np 1. This happens for rerun hvap
                        optimal_omp = [6, 4, 2] if n_gpu > 0 else [8, 6, 4, 2]
                        for i in optimal_omp:
                            if n_procs % (n_multi * i * 2) == 0:
                                n_thread = i
                                n_mpi = n_procs // n_thread
                                break

                    else:
                        # set n_mpi equal to n_multi
                        n_thread = n_procs // n_mpi

                cmd = re.sub('mpirun\s+-np\s+[0-9]+', '', cmd)  # remove mpirun -np xx
                cmd = re.sub('-ntomp\s+[0-9]+', '', cmd)  # remove -ntomp xx

                cmd = 'mpirun -np %i %s' % (n_mpi, cmd)  # add mpirun -np xx
                cmd += ' -multidir ' + ' '.join(dirs)  # add -multidir xx xx xx

                ### meaning of -gpu_id is changed in GROMACS 2018. Disable gpu assignment
                if n_gpu > 0 and self.majorversion == '2016':
                    cmd += ' -gpu_id ' + ''.join(map(str, range(n_gpu))) * (n_mpi // n_gpu) \
                           + ''.join(map(str, range(n_mpi % n_gpu)))  # add -gpu_id 01230123012
                if n_thread is not None:
                    cmd += ' -ntomp %i' % n_thread

            else:
                cmd = 'for i in %s; \ndo\n\tcd $i;\n\t%s &\ndone\nwait\n' % (
                ' '.join(dirs), cmd)  # do it in every directory
            return cmd

        commands_list: [[str]] = []
        n_group: int = math.ceil(len(dirs) / n_parallel)
        for n in range(n_group):
            multidir = dirs[n * n_parallel:(n + 1) * n_parallel]
            commands_multidir: [str] = []
            for j, dirname in enumerate(multidir):
                commands_multidir.append('dir%i=%s' % (j, dirname))  # replace full dir path with $dir1, $dir2 ...
            for cmd in commands:
                commands_multidir.append(replace_gpu_multidir_cmd(['$dir%i' % j for j in range(len(multidir))], cmd))
            commands_list.append(commands_multidir)
        return commands_list

    @staticmethod
    def modify_lj96_top(top):
        with open(top) as f:
            lines = f.read().splitlines()

        new_lines = []
        ATOMLINE = False
        for line in lines:
            if line.startswith(';') or line.strip() == '':
                new_lines.append(line)
                continue

            if line.find('[') != -1 and line.find('atoms') != -1:
                ATOMLINE = True
                new_lines.append(line)
                continue

            if ATOMLINE and line.find('[') != -1:
                ATOMLINE = False

            if ATOMLINE:
                words = line.strip().split()
                words[5] = words[0]
                tmp = ''
                for word in words:
                    tmp += ' %10s' % word
                new_lines.append(tmp)
            else:
                new_lines.append(line)
                continue

        with open(top, 'w') as f:
            f.write('\n'.join(new_lines))

    @staticmethod
    def modify_lj96_itp(itp):
        with open(itp) as f:
            lines = f.read().splitlines()
        new_lines = []
        for line in lines:
            if line.strip().split()[:3] == ['1', '2', 'yes']:
                new_lines.append(line + ' 9')
            else:
                new_lines.append(line)
        with open(itp, 'w') as f:
            f.write('\n'.join(new_lines))

    @staticmethod
    def modify_lj96_mdp(mdp):
        with open(mdp) as f:
            lines = f.read().splitlines()
        new_lines = []
        for line in lines:
            if line.strip().startswith('cutoff-scheme') or \
                    line.strip().startswith('cutoff_scheme') or \
                    line.strip().startswith('cutoffscheme') or \
                    line.strip().startswith('rlist'):
                pass
            else:
                new_lines.append(line)

            if line.strip().startswith('rvdw'):
                rvdw = line.split('=')[-1].strip()

        new_lines.append('\n; vdw table')
        new_lines.append('cutoff-scheme = group')
        new_lines.append('rlist = ' + rvdw)
        new_lines.append('vdwtype = user')
        with open(mdp, 'w') as f:
            f.write('\n'.join(new_lines))

    @staticmethod
    def modify_lj96(itp_files: [str], top_files: [str], mdp_files: [str], xvg_files: []):
        if itp_files is None:
            itp_files = list(filter(lambda x: x.endswith('.itp'), os.listdir('.')))
        for itp in itp_files:
            GMX.modify_lj96_itp(itp)

        if top_files is None:
            top_files = list(filter(lambda x: x.endswith('.top'), os.listdir('.')))
        for top in top_files:
            GMX.modify_lj96_top(top)

        if mdp_files is None:
            mdp_files = list(filter(lambda x: x.endswith('.mdp'), os.listdir('.')))
        for mdp in mdp_files:
            GMX.modify_lj96_mdp(mdp)

        for xvg in xvg_files:
            shutil.copy(os.path.join(GMX.TEMPLATE_DIR, 'table6-9.xvg'), xvg)

    def get_box_from_gro(self, gro):
        f = open(gro, 'r')
        box = f.readlines()[-1].split()
        return [float(box[0]), float(box[1]), float(box[2])]

    def get_volume_from_gro(self, gro):
        box = self.get_box_from_gro(gro)
        return box[0] * box[1] * box[2]

    def get_temperature_from_mdp(self, mdp):
        for line in open(mdp, 'r').readlines():
            if line.startswith('ref-t'):
                return float(line.split('=')[1])
        return None

    def current(self, trr, tpr, begin=0, end=None, skip=None, out=None, caf=False, select='System'):
        cmd = '%s -quiet -nobackup current -f %s -s %s -b %s' % (self.GMX_EXE, trr, tpr, str(begin))
        if end is not None:
            cmd += ' -e %s' % (str(end))
        if skip is not None:
            cmd += ' -skip %s' % (str(skip))
        if out is not None:
            cmd += ' -o %s' % (str(out))
        if caf:
            cmd += ' -caf'
        sp = Popen(cmd.split(), stdout=PIPE, stdin=PIPE, stderr=PIPE)
        out, err = sp.communicate(input=select.encode())
        out_str = ''
        for line in str(out).split('\\n'):
            if line not in ['', '"', '\'']:
                out_str += '%s\n' % (line)
        err_str = ''
        for line in str(err).split('\\n'):
            if line not in ['', '"', '\'']:
                err_str += '%s\n' % (line)

        return out_str, err_str

    def read_gmx_xvg(self, file=None):
        import pandas as pd
        if file is None:
            return None
        if file.endswith('caf.xvg'):
            info = pd.read_csv(file, sep='\s+', header=17)
            info.columns = ['time', 'acf', 'average', '#', '#']
            info = info.drop(['#'], axis=1)
            return info
        return None

    @staticmethod
    def get_n_mols_from_top(top):
        n1 = 10000  # the line start with "[ molecules ]"
        n2 = 100000  # the line end for "[ molecules ]"
        n_mols = 0
        for i, line in enumerate(open(top).readlines()):
            if line == '[ molecules ]\n':
                n1 = i
            if line.startswith('[') and i > n1:
                n2 = i
                break
            if n1 < i < n2 and len(line.split()) == 2:
                n_mols += int(line.split()[1])
        return n_mols
