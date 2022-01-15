# AIMS (Artifical Intelligence Molecular Simulation)
AIMS is a program designed to generate large-scale molecular property data sets using machine learning and computational chemistry techniques. 

It contains three main parts.
- Active Learning ([Gaussian Process Regression, Marginalized Graph Kernel](https://github.com/xiangyan93/Chem-Graph-Kernel-Machine)).
- High-throughput Quamtum Chemistry Calculation (GAUSSIAN) and classical Force-Field Molecular Dynamics Simulation (GROMACS).
- Machine Learning Prediction ([Direct Message Passing Neural Network](https://github.com/chemprop/chemprop)).

The workflow of AIMS:
<div align="center">
<p><img src="data/picture/AIMS.png" width="1000"/></p>
</div> 

The workflow of high-throughput calculation engine:
<div align="center">
<p><img src="data/picture/HTE.png" width="1000"/></p>
</div> 

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
- "monitor.py" need to run on a cluster with SLURM job manager to automatically submit jobs and
collect results.

## Usages
1. Submit molecules to the database.
   - Type SMILES of molecules.
   ```
   python3 submit.py --smiles CCCC CCCCC CCCCCC --files data/test.csv --features_generator rdkit_2d_normalized --heavy_atoms 4 19
   ```
2. Calculate the kernel matrix and save in data/kernel.pkl
   ```
   python3 calc_kernel.py --n_jobs 6
   ```
3. High-throughput QM calculation.
   ```
   python3 monitor.py --task qm_cv --partition cpu --n_cores 8 --n_jobs 8  --gaussian_exe $GAUSSIAN --n_conformer 1 --stop_uncertainty 0.3 --graph_kernel_type preCalc
   ```
4. High-throughput MD simulation.
   ```
   python3 monitor.py --task md_npt --partition gtx --n_cores 16 --n_hypercores 32 --n_gpu 2 --n_jobs 8 --packmol_exe $PACKMOL --dff_root $DFF --gmx_exe_analysis gmx_serial --gmx_exe_mdrun gmx_gpu --stop_uncertainty 0.3 --graph_kernel_type preCalc
   ```
5. Export simulation data.
   ```
   python3 export.py --property cp
   ```
6. Train D-MPNN
   ```
   python3 train.py --data_path cp.csv --dataset_type regression --save_dir ml-models/cp-dmpnn --split_type random --split_sizes 1.0 0.0 0.0 --metric mae --num_fold 1 --ensemble_size 1 --epochs 100 --save_preds --smiles_columns smiles --features_columns T --target_columns cp --features_generator rdkit_2d --num_workers 6 --aggregation sum
   ```
7. Predict D-MPNN
   ```
   python3 train.py --data_path cp_test_set.csv --preds_path cp_preds.csv --checkpoint_dir ml-models/cp-dmpnn --features_generator rdkit_2d --features_columns T
   ```
