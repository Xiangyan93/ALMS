#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy import create_engine, exists, and_, ForeignKey
from sqlalchemy import Column, Integer, Float, Text, Boolean, String, ForeignKey, UniqueConstraint
from rdkit import Chem
from simutools.forcefields.amber import AMBER
from simutools.simulator.program import GROMACS
from simutools.simulator.mol3d import Mol3D
from simutools.utils.utils import create_folder, create_missing_folders
from simutools.utils.rdkit import get_format_charge
from alms.utils import get_T_list_from_range
from .utils import *

CWD = os.path.dirname(os.path.abspath(__file__))
create_missing_folders(os.path.abspath(os.path.join(CWD, '..', '..', 'data', 'tmp')))
create_missing_folders(os.path.abspath(os.path.join(CWD, '..', '..', 'data', 'simulation', 'molecules')))
create_missing_folders(os.path.abspath(os.path.join(CWD, '..', '..', 'data', 'simulation', 'tasks_one_molecule')))
create_missing_folders(os.path.abspath(os.path.join(CWD, '..', '..', 'data', 'simulation', 'tasks_two_molecules')))

Base = declarative_base()
metadata = Base.metadata

db_file = 'sqlite:///%s' % os.path.abspath(os.path.join(CWD, '..', '..', 'data', 'alms.db'))

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
    BUILD = 1  # create all task input files
    PREPARED = 2  # create all job input files.
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

    def checkout(self, force_field: AMBER, simulator: Union[GROMACS]):
        cwd = os.getcwd()
        os.chdir(self.ms_dir)
        if not os.path.exists(f'{self.name}_ob.mol2'):
            force_field.checkout(smiles_list=[self.smiles], n_mol_list=[1], name_list=[self.resname],
                                 res_name_list=[self.resname], simulator=simulator)
        simulator.fix_charge('checkout.top')
        os.chdir(cwd)

    @property
    def ms_dir(self) -> str:
        ms_dir = os.path.join(CWD, '..', '..', 'data', 'simulation', 'molecules', str(self.id))
        create_folder(ms_dir)
        return os.path.abspath(ms_dir)

    @property
    def tt(self):
        """triple point temperature; melting point."""
        return json.loads(self.property_ml).get('tt')

    @property
    def tb(self):
        """boiling temperature"""
        return json.loads(self.property_ml).get('tb')

    @property
    def tc(self):
        """critical temperature"""
        return json.loads(self.property_ml).get('tc')

    @property
    def GetNumAtoms(self) -> int:
        mol = Chem.MolFromSmiles(self.smiles)
        return Chem.AddHs(mol).GetNumAtoms()

    @property
    def estimate_density(self) -> float:
        """A raw estimation of density. Only use in create the initial simulation box. unit: g/L"""
        return Mol3D(smiles=self.smiles).density

    @property
    def molwt(self) -> float:
        """"""
        return Mol3D(smiles=self.smiles).molwt

    @property
    def formal_charge(self) -> int:
        return get_format_charge(self.smiles)


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

    @property
    def name(self) -> str:
        return f'mol_{self.molecule_id}'

    @property
    def ms_dir(self) -> str:
        ms_dir = os.path.join(CWD, '..', '..', 'data', 'simulation', 'tasks_one_molecule', str(self.id))
        create_folder(ms_dir)
        return os.path.abspath(ms_dir)

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

    def create_jobs(self, task: Literal['qm_cv', 'md_npt'], n_conformer: int = 1,
                    T_min: float = None, T_max: float = None, n_T: int = None, P_list: List[float] = None):
        if task == 'qm_cv':
            for i in range(n_conformer):
                qm_cv = QM_CV(single_molecule_task_id=self.id, seed=i)
                add_or_query(qm_cv, ['single_molecule_task_id', 'seed'])
        elif task == 'md_npt':
            T_list = get_T_list_from_range(self.molecule.tc * T_min, self.molecule.tc * T_max, n_point=n_T)
            for T in T_list:
                for P in P_list:
                    md_npt = MD_NPT(single_molecule_task_id=self.id, T=T, P=P)
                    add_or_query(md_npt, ['single_molecule_task_id', 'T', 'P'])
        else:
            raise ValueError

    def delete_jobs(self, task: Literal['qm_cv', 'md_npt'], job_manager: Slurm = None):
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
    active = Column(Boolean, default=False)
    inactive = Column(Boolean, default=False)
    fail = Column(Boolean, default=False)

    molecules_id = Column(Text, unique=True)

    md_binding = relationship('MD_BINDING', back_populates='double_molecule_task')

    @property
    def name(self) -> str:
        return f'mol_{self.molecule_id_1}_mol_{self.molecule_id_2}'

    @property
    def ms_dir(self) -> str:
        ms_dir = os.path.join(CWD, '..', '..', 'data', 'simulation', 'tasks_two_molecules', str(self.id))
        create_folder(ms_dir)
        return os.path.abspath(ms_dir)

    @property
    def molecule_id_1(self) -> int:
        return int(self.molecules_id.split('_')[0])

    @property
    def molecule_id_2(self) -> int:
        return int(self.molecules_id.split('_')[1])

    @property
    def molecule_1(self):
        return session.query(Molecule).filter_by(id=self.molecule_id_1).first()

    @property
    def molecule_2(self):
        return session.query(Molecule).filter_by(id=self.molecule_id_2).first()

    @property
    def self_task(self) -> bool:
        return True if self.molecule_id_1 == self.molecule_id_2 else False

    def create_jobs(self, task: Literal['md_binding'], n_repeats: int = 1,
                    T_list: List[float] = None, P_list: List[float] = None):
        if task == 'md_binding':
            for T in T_list:
                for P in P_list:
                    for seed in range(n_repeats):
                        md_binding = MD_BINDING(double_molecule_task_id=self.id, T=T, P=P, seed=seed)
                        add_or_query(md_binding, ['double_molecule_task_id', 'T', 'P', 'seed'])
        else:
            raise ValueError

    def status(self, task: Literal['md_binding']) -> List[int]:
        if task == 'md_binding':
            return list(set([job.status for job in self.md_binding]))
        else:
            raise ValueError


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
    def name(self) -> str:
        return f'{self.single_molecule_task.name}_{self.__tablename__}_seed_{self.seed}'

    @property
    def ms_dir(self) -> str:
        ms_dir = os.path.join(self.single_molecule_task.ms_dir, self.__tablename__, f'seed_{self.seed}')
        create_missing_folders(ms_dir)
        return ms_dir

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
    def name(self) -> str:
        return f'{self.single_molecule_task.name}_{self.__tablename__}_T_{self.T}_P_{self.P}'

    @property
    def ms_dir(self) -> str:
        ms_dir = os.path.join(self.single_molecule_task.ms_dir, self.__tablename__, f'T_{self.T}_P_{self.P}')
        create_missing_folders(ms_dir)
        return ms_dir

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
    T = Column(Float)  # in K
    P = Column(Float)  # in bar
    commands_mdrun = Column(Text)
    commands_extend = Column(Text)
    sh_file = Column(Text)
    result = Column(Text)
    seed = Column(Integer, default=0)

    double_molecule_task_id = Column(Integer, ForeignKey('double_molecule_task.id'))
    double_molecule_task = relationship('DoubleMoleculeTask', back_populates='md_binding')

    @property
    def name(self) -> str:
        return f'{self.double_molecule_task.name}_{self.__tablename__}_T_{self.T}_P_{self.P}_seed_{self.seed}'

    @property
    def ms_dir(self) -> str:
        ms_dir = os.path.join(self.double_molecule_task.ms_dir,
                              self.__tablename__,
                              f'T_{self.T}_P_{self.P}_seed_{self.seed}')
        create_missing_folders(ms_dir)
        return ms_dir

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


metadata.create_all(engine)
