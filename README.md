# AIMS (Artifical Intelligence Molecular Simulation)
AIMS is a program for molecular property data sets. It contains three main parts.
- Active Learning ([Gaussian Process Regression, Marginalized Graph Kernel](https://github.com/xiangyan93/Chem-Graph-Kernel-Machine)).
- High-throughput Quamtum Chemistry Calculation and Force-Field Molecular Dynamics Simulation (GAUSSIAN, GROMACS).
- Machine Learning Prediction ([Direct Message Passing Neural Network](https://github.com/chemprop/chemprop)).

## Dependencies and Installation
- [GAUSSIAN](https://gaussian.com/gaussian16/) (Quantum Chemistry).
- [DFF](http://www.acc-sh.com/), 
[Packmol](http://leandro.iqm.unicamp.br/m3g/packmol/home.shtml), 
[GROMACS](https://manual.gromacs.org/documentation/) (Molecular Dynamics).
- Require GCC (7.*), NVIDIA Driver and CUDA toolkit(>=10.1) ([GraphDot](https://gitlab.com/yhtang/GraphDot)).
 
```
conda env create -f environment.yml
conda activate aims
```

## Usages
1. Submit molecules to the database.
   ```
   python3 submit.py --smiles CCCC CCCCC CCCCCC --features_generator rdkit_2d_normalized
   python3 submit.py --file data/smiles.csv --features_generator rdkit_2d_normalized --heavy_atoms 0 20
   ```
2. Select which molecules to be simulated using unsupervised active learning.
   ```
   python3 active_learning.py --stop_uncertainty 0.3
   ```
3. High-throughput QM simulation.
   ```
   python3 monitor.py --task qm_cv --partition cpu --n_cores 8 --n_jobs 8  --gaussian_exe $GAUSSIAN --n_conformer 1
   ```
4. High-throughput MD simulation.
   ```
   python3 monitor.py --task md_npt --partition gtx --n_cores 16 --n_hypercores 32 --n_gpu 2 --n_jobs 8 --packmol_exe $PACKMOL --dff_root $DFF --gmx_exe_analysis gmx_serial --gmx_exe_mdrun gmx_gpu
   ```
   