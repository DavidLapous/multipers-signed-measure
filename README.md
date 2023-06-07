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
It has been tested with python 3.11 on Linux (gcc12.2) and Macos (clang14).

## How to use : tutorial notebooks
We provide a few notebooks, which explains, in different scenarios, how to use our library. **Take a look at them !** They are in the tutorial folder.


## References
This library is a fork of the [MMA](https://github.com/DavidLapous/multipers) library, which handles multiparameter simplicial complex structures, aswell as approximation of multiparameter persistence modules *via* interval decompositions.

The usage of signed measures for the vectorization of multiparameter persistence modules was introduced in [Stable Vectorization of Multiparameter Persistent Homology using Signed Barcodes as Measures](https://arxiv.org/abs/2306.03801). They are fast, and easily usable in a machine learning context.

2-parameter edge collapses are realized using [filtration_domination](https://github.com/aj-alonso/filtration_domination/).


## Authors
[David Loiseaux](https://www-sop.inria.fr/members/David.Loiseaux/index.html), [Luis Scoccola](https://luisscoccola.com/) 
(Möbius inversion in python, degree-rips using [persistable](https://github.com/LuisScoccola/persistable) and [RIVET](https://github.com/rivetTDA/rivet/)), [Mathieu Carrière](https://www-sop.inria.fr/members/Mathieu.Carriere/) (Sliced Wasserstein).

## Contributions
Hannah Schreiber

Feel free to contribute, report a bug on a pipeline, or ask for documentation by opening an issue.<br>
**Any** contribution is welcome.
