# AIMS

## Installation
Require GCC (7.*), NVIDIA Driver and CUDA toolkit(>=10.1).  
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
3. Run QM simulation of selected molecules using active learning.
   ```
   python3 monitor.py --task qm_cv --partition cpu --n_cores 8 --n_jobs 8  --gaussian_exe $GAUSSIAN --n_conformer 1
   ```
4. Run MD simulation of selected molecules using active learning.
   ```
   python3 monitor.py --task md_npt --partition cpu --n_cores 8 --n_jobs 8 --packmol_exe $PACKMOL --dff_root $DFF --gmx_exe_analysis $gmx_serial --gmx_exe_mdrun $gmx_gpu
   ```
   