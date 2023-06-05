# Multiparameter Persistence using Signed Measures

## Installation
This python library has a C++ backend which needs to be installed. 

### Dependencies
It has a few dependencies that can be installed with, e.g., conda.
```sh
conda create -n python311
conda activate python311
conda install python=3.11 boost tbb tbb-devel numpy matplotlib gudhi scikit-learn cython sympy tqdm cycler typing shapely numba -c conda-forge
pip install filtration-domination
```

### Compile-install
The following installs the `multipers` library
```sh
pip install .
```
## How to use : tutorial notebooks
We provide a few notebooks, which explains, in different scenarios, how to use our library. **Take a look at them !** They are in the tutorial folder.


## References
This library is a fork of the [MMA](https://github.com/DavidLapous/multipers) library, which handles multiparameter simplicial complex structures, aswell as approximation of multiparameter persistence modules *via* interval decompositions.

**Signed measures** are introduced in [Stable Vectorization of Multiparameter Persistent Homology using Signed Barcodes as Measures](). They are fast, and easily usable in a machine learning context.

2-parameter edge collapses are realized using [filtration_domination](https://github.com/aj-alonso/filtration_domination/).


## Authors
David Loiseaux, Luis Scoccola (Möbius inversion in python), Mathieu Carrière (Sliced Wasserstein). 

## Contributions
Hannah Schreiber

Feel free to contribute, report a bug on a pipeline, or ask for documentation by opening an issue. **Any** contribution is welcome.