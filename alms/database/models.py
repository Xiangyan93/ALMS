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

db_file = 'sqlite:///%s' % os.path.join(CWD, '..', '..', 'data', 'alms.db')

engine = create_engine(db_file, echo=False)
Session = sessionmaker(engine)
session = Session()


def add_or_query(row, keys: List[str]):
    """ Add a new item into the database if it is not existed. Otherwise, query the existed one.

    Parameters
    ----------
    row: A row of data.
    keys: The unique keys of the data.

    Returns
    -------
    A row of data.
    """
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
    smiles = Column(Text, unique=True)
    name = Column(Text, unique=True)
    resname = Column(Text, unique=True)
    property_ml = Column(Text)
    tag = Column(Text)
    info = Column(Text)
    single_molecule_task = relationship('SingleMoleculeTask', uselist=False, back_populates='molecule')

    def update_dict(self, attr: str, p_dict: Dict):
        update_dict(self, attr, p_dict)

    @property
    def ms_dir(self) -> str:
        ms_dir = os.path.join(CWD, '..', '..', 'data', 'ms', 'molecules', str(self.id))
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


class SingleMoleculeTask(Base):
    __tablename__ = 'single_molecule_task'
    id = Column(Integer, primary_key=True)
    active = Column(Boolean, default=False)
    inactive = Column(Boolean, default=False)
    fail = Column(Boolean, default=False)
    molecule_id = Column(Integer, ForeignKey('molecule.id'))
    molecule = relationship('Molecule', back_populates='single_molecule_task')
    qm_cv = relationship('QM_CV', back_populates='single_molecule_task')
    md_npt = relationship('MD_NPT', back_populates='single_molecule_task')

    testset = Column(Boolean, default=False)
    # tag = Column(Text)
    # property_ms = Column(Text)

    @property
    def ms_dir(self) -> str:
        ms_dir = os.path.join(CWD, '..', '..', 'data', 'ms', 'task_1molecule', str(self.id))
        if not os.path.exists(ms_dir):
            os.mkdir(ms_dir)
        return ms_dir

    def set_status(self, attr: str, status: int):
        for job in getattr(self, attr):
            job.status = status

    def status(self, task: Literal['qm_cv', 'md_npt']) -> List[int]:
        if task == 'qm_cv':
            return list(set([job.status for job in self.qm_cv]))
        elif task == 'md_npt':
            return list(set([job.status for job in self.md_npt]))
        else:
            raise ValueError
    # functions for qm_cv

    def create_jobs(self, task: Literal['qm_cv', 'md_npt'], n_conformer: int = 1,
                    T_min: float = None, T_max: float = None, n_T: int = None, P_list: List[float] = None):
        if task == 'qm_cv':
            for i in range(n_conformer):
                qm_cv = QM_CV(molecule_id=self.id, seed=i)
                add_or_query(qm_cv, ['molecule_id', 'seed'])
        elif task == 'md_npt':
            T_list = get_T_list_from_range(self.tc * T_min, self.tc * T_max, n_point=n_T)
            for T in T_list:
                for P in P_list:
                    md_npt = MD_NPT(molecule_id=self.id, T=T, P=P)
                    add_or_query(md_npt, ['molecule_id', 'T', 'P'])

    def reset_jobs(self, task: Literal['qm_cv', 'md_npt'], job_manager: Slurm = None):
        jobs = self.qm_cv if task == 'qm_cv' else self.md_npt
        for job in jobs:
            job.delete(job_manager)
        try:
            shutil.rmtree(os.path.join(self.ms_dir, task))
        except:
            pass


class DoubleMoleculeTask(Base):
    __tablename__ = 'double_molecule_task'
    id = Column(Integer, primary_key=True)
    al_status = Column(Boolean, default=False)
    fail = Column(Boolean, default=False)
    md_binding = relationship('MD_BINDING', back_populates='double_molecule_task')

    @property
    def ms_dir(self) -> str:
        ms_dir = os.path.join(CWD, '..', '..', 'data', 'ms', 'task_2molecules', str(self.id))
        if not os.path.exists(ms_dir):
            os.mkdir(ms_dir)
        return ms_dir


class QM_CV(Base):
    __tablename__ = 'qm_cv'
    id = Column(Integer, primary_key=True)
    status = Column(Integer, default=Status.STARTED)
    seed = Column(Integer, default=0)
    commands = Column(Text)
    sh_file = Column(Text)
    result = Column(Text)

    single_molecule_task_id = Column(Integer, ForeignKey('single_molecule_task.id'))
    single_molecule_task = relationship('SingleMoleculeTask', back_populates='qm_cv')

    @property
    def ms_dir(self) -> str:
        ms_dir = os.path.join(self.single_molecule_task.ms_dir, self.__tablename__)
        if not os.path.exists(ms_dir):
            os.mkdir(ms_dir)
        ms_dir = os.path.join(self.single_molecule_task.ms_dir, self.__tablename__, 'conformer_%d' % self.seed)
        if not os.path.exists(ms_dir):
            os.mkdir(ms_dir)
        return ms_dir

    @property
    def name(self) -> str:
        return 'aims_qm_cv_%d' % self.single_molecule_task.id

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

    single_molecule_task_id = Column(Integer, ForeignKey('single_molecule_task.id'))
    single_molecule_task = relationship('SingleMoleculeTask', back_populates='md_npt')

    @property
    def ms_dir(self) -> str:
        ms_dir = os.path.join(self.single_molecule_task.ms_dir, self.__tablename__)
        if not os.path.exists(ms_dir):
            os.mkdir(ms_dir)
        ms_dir = os.path.join(self.single_molecule_task.ms_dir, self.__tablename__, '%d_%d' % (self.T, self.P))
        if not os.path.exists(ms_dir):
            os.mkdir(ms_dir)
        return ms_dir

    @property
    def name(self) -> str:
        return 'aims_md_npt_ID%d_T%d_P%d' % (self.single_molecule_task.id, self.T, self.P)

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


class MD_BINDING(Base):
    __tablename__ = 'md_binding'
    id = Column(Integer, primary_key=True)
    status = Column(Integer, default=Status.STARTED)
    seed = Column(Integer, default=0)
    commands = Column(Text)
    sh_file = Column(Text)
    result = Column(Text)

    double_molecule_task_id = Column(Integer, ForeignKey('double_molecule_task.id'))
    double_molecule_task = relationship('DoubleMoleculeTask', back_populates='md_binding')


metadata.create_all(engine)
