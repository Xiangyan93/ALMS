# AIMS

## Installation
Require GCC (7.*), NVIDIA Driver and CUDA toolkit(>=10.1).  
```
conda env create -f environment.yml
conda activate graphdot
```

## Usages
1. Submit molecules to the database.
   ```
   python3 submit.py --smiles CCCC CCCCC CCCCCC --features_generator rdkit_2d_normalized
   python3 submit.py --file data/smiles.csv --features_generator rdkit_2d_normalized --heavy_atoms 0 20
   ```
2. Select which molecules to be simulated using unsupervised active learning.
   ```
   python3 active_learning.py --stop_uncertainty 0.5
   ```
3. Run QM simulation of selected molecules using active learning.
   ```
   python3 monitor.py --task qm_cv --partition cpu --GAUSSIAN_BIN $GAUSSIAN_BIN --n_conformer 1 --n_cpu 8 --n_jobs 8
   ```