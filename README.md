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
   python3 monitor.py --task qm_cv --partition cpu --GAUSSIAN_EXE $GAUSSIAN_BIN --n_conformer 1 --n_cores 8 --n_jobs 8
   ```