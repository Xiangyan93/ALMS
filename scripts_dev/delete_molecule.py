#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tap import Tap
from alms.database.models import *
from tqdm import tqdm


class Args(Tap):
    name: str = None
    """The name of the molecule."""
    id_range: Tuple[int, int] = None
    """The range of molecule ids to delete."""


def main(args: Args):
    job_manager = Slurm()
    job_manager.update_stored_jobs()
    mols = session.query(Molecule)
    delete_mols = []
    if args.name is not None:
        delete_mols.append(mols.filter_by(name=args.name).first())
    if args.id_range is not None:
        for mol in mols:
            if args.id_range[0] <= mol.id <= args.id_range[1]:
                delete_mols.append(mol)
    delete_mols_id = [mol.id for mol in delete_mols]
    for task in tqdm(session.query(DoubleMoleculeTask), total=session.query(DoubleMoleculeTask).count()):
        if task.molecule_id_1 in delete_mols_id or task.molecule_id_2 in delete_mols_id:
            task.delete(job_manager=job_manager)
            session.commit()
    for mol in delete_mols:
        if mol.single_molecule_task is not None:
            mol.single_molecule_task.delete(job_manager=job_manager)
        session.delete(mol)
        session.commit()


if __name__ == '__main__':
    main(args=Args().parse_args())

