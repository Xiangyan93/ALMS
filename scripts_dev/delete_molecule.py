#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tap import Tap
from alms.database.models import *


class Args(Tap):
    name: str
    """The name of the molecule."""


def main(args: Args):
    job_manager = Slurm()
    job_manager.update_stored_jobs()
    mol = session.query(Molecule).filter_by(name=args.name).first()
    for task in session.query(DoubleMoleculeTask):
        if task.molecule_id_1 == mol.id or task.molecule_id_2 == mol.id:
            task.delete(job_manager=job_manager)
    session.delete(mol)
    session.commit()


if __name__ == '__main__':
    main(args=Args().parse_args())

