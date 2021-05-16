#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
CWD = os.path.dirname(os.path.abspath(__file__))
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import re
import json
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


class Status:
    STARTED = 0  # create task.
    PREPARED = 1  # create all input files.
    SUBMITED = 2  # submit slurm jobs.
    DONE = 3  # slurm jobs finished.
    ANALYZED = 4  # read the log file and extract results successfully.
    NOT_CONVERGED = 5  # the simultion is not converged, need to be extended.
    EXTENDED = 6  #
    FAILED = -1  # failed task.


class Molecule(Base):
    __tablename__ = 'molecule'
    id = Column(Integer, primary_key=True)
    smiles = Column(String(255), unique=True)
    active_learning = Column(Boolean, default=False)
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

    def features_(self, features_generator: str = None):
        return json.loads(self.features).get(features_generator)

    def update_dict(self, attr: str, p_dict: Dict):
        update_dict(self, attr, p_dict)

    def create_qm_cv(self, n_conformer: int = 1):
        for i in range(n_conformer):
            qm_cv = QM_CV(seed=i, molecule_id=self.id)
            add_or_query(qm_cv, ['molecule_id', 'seed'])


class QM_CV(Base):
    __tablename__ = 'qm_cv'
    id = Column(Integer, primary_key=True)

    status = Column(Integer, default=Status.STARTED)
    seed = Column(Integer, default=0)
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
        return 'aims_qm_cv_%d' % self.id

    def update_dict(self, attr: str, p_dict: Dict):
        update_dict(self, attr, p_dict)

    def update_list(self, attr: str, p_list: List):
        update_list(self, attr, p_list)


class MD_NPT(Base):
    __tablename__ = 'md_npt'
    id = Column(Integer, primary_key=True)

    status = Column(Integer, default=Status.STARTED)
    T = Column(Integer)  # in K
    P = Column(Integer)  # in bar
    sh_file = Column(Text)

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

    def update_dict(self, attr: str, p_dict: Dict):
        update_dict(self, attr, p_dict)

    n_components = Column(Integer)
    smiles_list = Column(Text)
    n_mol_list = Column(Text, nullable=True)
    procedure = Column(String(200))
    t_list = Column(Text)
    p_list = Column(Text)
    n_mol_ratio = Column(Text)
    # name = Column(String(200), default=random_string)
    #stage = Column(Integer, default=Compute.Stage.SUBMITTED)
    #status = Column(Integer, default=Compute.Status.DONE)
    commands = Column(Text, nullable=True)
    remark = Column(Text, nullable=True)
    post_result = Column(Text, nullable=True)
    atom_type = Column(Text)


metadata.create_all(engine)