#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
CWD = os.path.dirname(os.path.abspath(__file__))
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import (
    sessionmaker,
    relationship
)
from sqlalchemy import (
    create_engine,
    exists,
    and_,
    ForeignKey
)
from sqlalchemy import (
    Column, Integer, Float, Text, Boolean, String, ForeignKey, UniqueConstraint,
)
from ..aimstools.utils import get_T_list_from_range
from .utils import *


Base = declarative_base()
metadata = Base.metadata

db_file = 'sqlite:///%s' % os.path.join(CWD, '..', '..', 'data', 'aims.db')

engine = create_engine(db_file, echo=False)
Session = sessionmaker(engine)
session = Session()


def add_or_query(row, keys: List[str]):
    filter_dict = dict()
    for key in keys:
        filter_dict[key] = row.__dict__[key]

    result = session.query(row.__class__).filter_by(**filter_dict).first()
    if not result:
        session.add(row)
        session.flush()
        return row
    else:
        return result


class Status:
    STARTED = 0  # create task.
    BUILD = 1
    PREPARED = 2  # create all input files.
    SUBMITED = 3  # submit slurm jobs.
    DONE = 4  # slurm jobs finished.
    ANALYZED = 5  # read the log file and extract results successfully.
    NOT_CONVERGED = 6  # the simultion is not converged, need to be extended.
    EXTENDED = 7  # the not-converged simulation is extended, need to submit slurm jobs.
    MIXED = 8
    FAILED = -1  # failed task.


class Molecule(Base):
    __tablename__ = 'molecule'
    id = Column(Integer, primary_key=True)
    smiles = Column(String(255), unique=True)
    active = Column(Boolean, default=False)
    inactive = Column(Boolean, default=False)
    fail = Column(Boolean, default=False)
    testset = Column(Boolean, default=False)
    tag = Column(Text)
    features = Column(Text)
    property_exp = Column(Text)
    property_ml = Column(Text)
    property_ms = Column(Text)
    qm_cv = relationship('QM_CV', back_populates='molecule')
    md_npt = relationship('MD_NPT', back_populates='molecule')

    @property
    def ms_dir(self) -> str:
        ms_dir = os.path.join(CWD, '..', '..', 'data', 'ms', str(self.id))
        if not os.path.exists(ms_dir):
            os.mkdir(ms_dir)
        return ms_dir

    @property
    def tt(self):
        return json.loads(self.property_ml).get('tt')

    @property
    def tb(self):
        return json.loads(self.property_ml).get('tb')

    @property
    def tc(self):
        return json.loads(self.property_ml).get('tc')

    def features_(self, features_generator: str = None):
        return json.loads(self.features).get(features_generator)

    def update_dict(self, attr: str, p_dict: Dict):
        update_dict(self, attr, p_dict)

    def set_status(self, attr: str, status: int):
        for job in getattr(self, attr):
            job.status = status

    # functions for qm_cv
    @property
    def status_qm_cv(self) -> List[int]:
        return list(set([job.status for job in self.qm_cv]))

    def create_qm_cv(self, n_conformer: int = 1):
        for i in range(n_conformer):
            qm_cv = QM_CV(molecule_id=self.id, seed=i)
            add_or_query(qm_cv, ['molecule_id', 'seed'])

    def reset_qm_cv(self, job_manager: Slurm = None):
        for job in self.qm_cv:
            job.delete(job_manager)
        try:
            shutil.rmtree(os.path.join(self.ms_dir, 'qm_cv'))
        except:
            pass
    # functions for qm_cv

    # functions for md_npt
    @property
    def status_md_npt(self) -> List[int]:
        return list(set([job.status for job in self.md_npt]))

    def create_md_npt(self, T_min: float, T_max: float, n_T: int, P_list: List[float]):
        T_list = get_T_list_from_range(self.tc * T_min, self.tc * T_max, n_point=n_T)
        for T in T_list:
            for P in P_list:
                md_npt = MD_NPT(molecule_id=self.id, T=T, P=P)
                add_or_query(md_npt, ['molecule_id', 'T', 'P'])

    def reset_md_npt(self, job_manager: Slurm = None):
        for job in self.md_npt:
            job.delete(job_manager)
        try:
            shutil.rmtree(os.path.join(self.ms_dir, 'md_npt'))
        except:
            pass
    # functions for md_npt


class QM_CV(Base):
    __tablename__ = 'qm_cv'
    id = Column(Integer, primary_key=True)

    status = Column(Integer, default=Status.STARTED)
    seed = Column(Integer, default=0)
    commands = Column(Text)
    sh_file = Column(Text)
    result = Column(Text)

    molecule_id = Column(Integer, ForeignKey('molecule.id'))
    molecule = relationship('Molecule', back_populates='qm_cv')

    @property
    def ms_dir(self) -> str:
        ms_dir = os.path.join(self.molecule.ms_dir, self.__tablename__)
        if not os.path.exists(ms_dir):
            os.mkdir(ms_dir)
        ms_dir = os.path.join(self.molecule.ms_dir, self.__tablename__, 'conformer_%d' % self.seed)
        if not os.path.exists(ms_dir):
            os.mkdir(ms_dir)
        return ms_dir

    @property
    def name(self) -> str:
        return 'aims_qm_cv_%d' % self.molecule.id

    @property
    def slurm_name(self) -> Optional[str]:
        sh_file = json.loads(self.sh_file)
        if sh_file:
            assert sh_file[-1].endswith('.sh')
            return sh_file[-1].split('/')[-1][:-3]
        else:
            return None

    def update_dict(self, attr: str, p_dict: Dict):
        update_dict(self, attr, p_dict)

    def update_list(self, attr: str, p_list: List):
        update_list(self, attr, p_list)

    def delete(self, job_manager: Slurm = None):
        delete_job(job=self, session=session, job_manager=job_manager)


class MD_NPT(Base):
    __tablename__ = 'md_npt'
    id = Column(Integer, primary_key=True)

    status = Column(Integer, default=Status.STARTED)
    T = Column(Float)  # in K
    P = Column(Float)  # in bar
    commands_mdrun = Column(Text)
    commands_extend = Column(Text)
    sh_file = Column(Text)
    result = Column(Text)

    molecule_id = Column(Integer, ForeignKey('molecule.id'))
    molecule = relationship('Molecule', back_populates='md_npt')

    @property
    def ms_dir(self) -> str:
        ms_dir = os.path.join(self.molecule.ms_dir, self.__tablename__)
        if not os.path.exists(ms_dir):
            os.mkdir(ms_dir)
        ms_dir = os.path.join(self.molecule.ms_dir, self.__tablename__, '%d_%d' % (self.T, self.P))
        if not os.path.exists(ms_dir):
            os.mkdir(ms_dir)
        return ms_dir

    @property
    def name(self) -> str:
        return 'aims_md_npt_ID%d_T%d_P%d' % (self.molecule.id, self.T, self.P)

    @property
    def slurm_name(self) -> Optional[str]:
        sh_file = json.loads(self.sh_file)
        if sh_file:
            assert sh_file[-1].endswith('.sh')
            return sh_file[-1].split('/')[-1][:-3]
        else:
            return None

    @property
    def mdrun_times(self) -> int:
        log = os.path.join(self.ms_dir, 'npt.log')
        if not os.path.exists(log):
            return 0
        f = open(log, 'r')
        n = 0
        for line in f.readlines():
            if line.startswith('Started mdrun'):
                n += 1
        return n

    def update_dict(self, attr: str, p_dict: Dict):
        update_dict(self, attr, p_dict)

    def update_list(self, attr: str, p_list: List):
        update_list(self, attr, p_list)

    def delete(self, job_manager: Slurm = None):
        delete_job(job=self, session=session, job_manager=job_manager)


metadata.create_all(engine)
