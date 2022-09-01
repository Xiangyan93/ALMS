#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
CWD = os.path.dirname(os.path.abspath(__file__))
import subprocess
import sys
from subprocess import PIPE, Popen

from ...utils import random_string


class DffError(Exception):
    pass


class DFF:
    TEMPLATE_DIR = os.path.join(CWD, 'template')
    '''
    wrappers for DFF
    '''
    pass

    def __init__(self, dff_root: str, default_db: str = 'TEAMFF', default_table: str = 'MGI'):
        self.DFF_ROOT = os.path.abspath(dff_root)
        if sys.platform == 'darwin':
            self.DFF_BIN_DIR = os.path.join(self.DFF_ROOT, 'bin64m')
        elif sys.platform.startswith('linux'):
            self.DFF_BIN_DIR = os.path.join(self.DFF_ROOT, 'bin64x')
        elif sys.platform.startswith('win'):
            self.DFF_BIN_DIR = os.path.join(self.DFF_ROOT, 'bin32w')
        else:
            raise DffError('Unsupported platform')

        self.DFFJOB_BIN = os.path.join(self.DFF_BIN_DIR, 'dffjob.exe')
        self.DFFEXP_BIN = os.path.join(self.DFF_BIN_DIR, 'dffexp.exe')
        self.DFFFIT_BIN = os.path.join(self.DFF_BIN_DIR, 'dfffit.exe')
        self.default_db = default_db
        self.default_table = default_table

    def convert_model_to_msd(self, model, msd_out, dfi_name='convert'):
        dfi = open(os.path.join(DFF.TEMPLATE_DIR, 't_convert.dfi')).read()
        dfi = dfi.replace('%MODEL%', model).replace('%OUTPUTMODEL%', msd_out)
        with open(dfi_name + '.dfi', 'w') as f:
            f.write(dfi)
        sp = subprocess.Popen([self.DFFJOB_BIN, dfi_name], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        out, err = sp.communicate()
        if err.decode() != '':
            raise DffError('Convert failed: %s' % err.decode())

    def checkout(self, models: [str], db=None, table=None, ppf_out=None, dfi_name='checkout'):
        if db is None:
            db = os.path.join(self.DFF_ROOT, 'database', '%s.dffdb' % self.default_db)
        if table is None:
            table = self.default_table
        if ppf_out is None:
            ppf_out = table + '.ppf'
        model_path = []
        for model in models:
            model_path.append(os.path.abspath(model))
        db = os.path.abspath(db)
        ppf_out = os.path.abspath(ppf_out)
        dfi = open(os.path.join(DFF.TEMPLATE_DIR, 't_checkout.dfi')).read()
        dfi = dfi.replace('%DATABASE%', db).replace('%TABLE%', table).replace('%MODELS%', '\n'.join(model_path)) \
            .replace('%OUTPUT%', ppf_out).replace('%LOG%', os.path.abspath('checkout.dfo'))
        with open(dfi_name + '.dfi', 'w') as f:
            f.write(dfi)
        sp = subprocess.Popen([self.DFFJOB_BIN, dfi_name], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        out, err = sp.communicate()
        if err.decode() != '':
            raise DffError('Checkout failed: %s' % err.decode())

    def set_formal_charge(self, models: [str], dfi_name='setfc'):
        model_path = []
        for model in models:
            model_path.append(os.path.abspath(model))
        log = os.path.abspath(dfi_name + '.dfo')
        dfi = open(os.path.join(DFF.TEMPLATE_DIR, 't_set_formal_charge.dfi')).read()
        dfi = dfi.replace('%MODELS%', '\n'.join(model_path)).replace('%LOG%', log)
        with open(dfi_name + '.dfi', 'w') as f:
            f.write(dfi)
        sp = subprocess.Popen([self.DFFJOB_BIN, dfi_name], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        out, err = sp.communicate()
        if err.decode() != '':
            raise DffError('Set formal charge failed: %s' % err.decode())

    def typing(self, models: [str], rule=None, dfi_name='typing'):
        model_path = []
        for model in models:
            model_path.append(os.path.abspath(model))
        if rule is None:
            rule = os.path.join(self.DFF_ROOT, 'database',
                                '%s.ref/%s/%s.ext' % (self.default_db, self.default_table, self.default_table))
        else:
            rule = os.path.abspath(rule)
        log = os.path.abspath(dfi_name + '.dfo')
        dfi = open(os.path.join(DFF.TEMPLATE_DIR, 't_typing.dfi')).read()
        dfi = dfi.replace('%MODELS%', '\n'.join(model_path)).replace('%RULE%', rule).replace('%LOG%', log)
        with open(dfi_name + '.dfi', 'w') as f:
            f.write(dfi)
        sp = subprocess.Popen([self.DFFJOB_BIN, dfi_name], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        out, err = sp.communicate()
        if err.decode() != '':
            raise DffError('Typing failed: %s' % err.decode())

    def set_charge(self, models: [str], ppf, dfi_name='set_charge'):
        model_path = []
        for model in models:
            model_path.append(os.path.abspath(model))
        ppf = os.path.abspath(ppf)
        log = os.path.abspath(dfi_name + '.dfo')
        dfi = open(os.path.join(DFF.TEMPLATE_DIR, 't_set_charge.dfi')).read()
        dfi = dfi.replace('%MODELS%', '\n'.join(model_path)).replace('%PPF%', ppf).replace('%LOG%', log)
        with open(dfi_name + '.dfi', 'w') as f:
            f.write(dfi)
        sp = subprocess.Popen([self.DFFJOB_BIN, dfi_name], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        out, err = sp.communicate()
        if err.decode() != '':
            raise DffError('Set charge failed: %s' % err.decode())

    def export_lammps(self, model, ppf, data_out='data', lmp_out='in.lmp', dfi_name='export_lammps'):
        model = os.path.abspath(model)
        ppf = os.path.abspath(ppf)
        data_out = os.path.abspath(data_out)
        lmp_out = os.path.abspath(lmp_out)

        fftype = self.get_ff_type_from_ppf(ppf)

        dfi = open(os.path.join(DFF.TEMPLATE_DIR, 't_export_lammps.dfi')).read()
        dfi = dfi.replace('%ROOT%', self.DFF_ROOT).replace('%MODEL%', model).replace('%PPF%', ppf) \
            .replace('%TEMPLATE%', 'LAMMPS.Opt').replace('%FFTYPE%', fftype) \
            .replace('%DATAFILE%', data_out).replace('%INFILE%', lmp_out)
        with open(dfi_name + '.dfi', 'w') as f:
            f.write(dfi)
        sp = subprocess.Popen([self.DFFEXP_BIN, dfi_name], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        out, err = sp.communicate()
        if err.decode() != '':
            raise DffError('Export failed: %s' % err.decode())

    def export_gmx(self, model, ppf, gro_out='conf.gro', top_out='topol.top', mdp_out='grompp.mdp',
                   dfi_name='export_gmx'):
        model = os.path.abspath(model)
        ppf = os.path.abspath(ppf)
        gro_out = os.path.abspath(gro_out)
        top_out = os.path.abspath(top_out)
        mdp_out = os.path.abspath(mdp_out)
        itp_out = top_out[:-4] + '.itp'

        fftype = self.get_ff_type_from_ppf(ppf)

        dfi = open(os.path.join(DFF.TEMPLATE_DIR, 't_export_gmx.dfi')).read()
        dfi = dfi.replace('%ROOT%', self.DFF_ROOT).replace('%MODEL%', model).replace('%PPF%', ppf) \
            .replace('%TEMPLATE%', 'GROMACS.Opt').replace('%FFTYPE%', fftype) \
            .replace('%GROFILE%', gro_out).replace('%TOPFILE%', top_out).replace('%MDPFILE%', mdp_out) \
            .replace('%ITPFILE%', itp_out)
        with open(dfi_name + '.dfi', 'w') as f:
            f.write(dfi)
        sp = Popen([self.DFFEXP_BIN, dfi_name], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        out, err = sp.communicate()
        if err.decode() != '':
            raise DffError('Export failed: %s' % err.decode())

    def build_box_after_packmol(self, models: [str], numbers: [int], msd_out, mol_corr,
                                box_size: [float] = None, length: float = None, dfi_name='set_corr'):
        if box_size != None:
            if len(box_size) != 3:
                raise DffError('Invalid box size')
            else:
                box = box_size
        elif length != None:
            box = [length, length, length]
        else:
            raise DffError('Box size needed')

        model_list_str = ''
        for i, model in enumerate(models):
            model = os.path.abspath(model)
            if i == 0:
                model_list_str += 'MOL=%s' % model
            else:
                model_list_str += '\n    MOL=%s' % model
        number_list_str = ' '.join(map(str, numbers))

        msd_tmp = os.path.abspath(random_string() + '.msd')
        msd_out = os.path.abspath(msd_out)
        mol_corr = os.path.abspath(mol_corr)
        dfi = open(os.path.join(DFF.TEMPLATE_DIR, 't_packmol.dfi')).read()
        dfi = dfi.replace('%MODEL_LIST%', model_list_str).replace('%NUMBER_LIST%', number_list_str) \
            .replace('%MSD_TMP%', msd_tmp).replace('%OUT%', msd_out) \
            .replace('%CORRMOL%', mol_corr).replace('%PBC%', ' '.join(map(str, box)))
        with open(dfi_name + '.dfi', 'w') as f:
            f.write(dfi)
        sp = Popen([self.DFFJOB_BIN, dfi_name], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        out, err = sp.communicate()
        if err.decode() != '':
            raise DffError('Build failed: %s' % err.decode())

        try:
            os.remove(msd_tmp)
        except:
            pass

    def fit_torsion(self, qmd, msd, ppf_in, ppf_out, torsion, dfi_name='fit_torsion'):
        'TORS C1 C2 C3 C4 1000.0 180.0 15.0 12'
        fftype = self.get_ff_type_from_ppf(ppf_in)
        dfi = open(os.path.join(DFF.TEMPLATE_DIR, 't_fit_torsion.dfi')).read()
        dfi = dfi.replace('%ROOT%', self.DFF_ROOT).replace('%QMD%', qmd).replace('%MSD%', msd) \
            .replace('%FFTYPE%', fftype).replace('%PPF_IN%', ppf_in).replace('%PPF_OUT%', ppf_out) \
            .replace('%TORSION%', torsion)
        with open(dfi_name + '.dfi', 'w') as f:
            f.write(dfi)
        sp = Popen([self.DFFFIT_BIN, dfi_name], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        out, err = sp.communicate()
        if err.decode() != '':
            raise DffError('Fit torsion failed: %s' % err.decode())

    @staticmethod
    def get_ff_type_from_ppf(ppf):
        with open(ppf) as f_ppf:
            for line in f_ppf:
                if line.startswith('#PROTOCOL'):
                    fftype = line.split('=')[-1].strip()
                    return fftype
        raise DffError('Unknown FF Type: %s' % ppf)
