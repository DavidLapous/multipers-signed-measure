# Multiparameter Persistence using Signed Measures
Scikit-style multiparameter persistent homology python library, using signed measure and their representation for machine learning. 

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
It has been tested with python 3.11 on Linux (gcc12.2) and Macos (clang14-clang16). If the build fails (on macos) see a fix at the end of the readme. 

## How to use : tutorial notebooks
We provide a few notebooks, which explains, in different scenarios, how to use our library. **Take a look at them !** They are in the tutorial folder.


## References
This library is a fork of the [MMA](https://github.com/DavidLapous/multipers) library, which handles multiparameter simplicial complex structures, aswell as approximation of multiparameter persistence modules *via* interval decompositions [[arxiv](https://arxiv.org/abs/2206.02026)].

The usage of signed measures for the vectorization of multiparameter persistence modules was introduced in [Stable Vectorization of Multiparameter Persistent Homology using Signed Barcodes as Measures](https://arxiv.org/abs/2306.03801). They are fast, and easily usable in a machine learning context.

2-parameter edge collapses are realized using [filtration_domination](https://github.com/aj-alonso/filtration_domination/).


## Authors
[David Loiseaux](https://www-sop.inria.fr/members/David.Loiseaux/index.html), [Luis Scoccola](https://luisscoccola.com/) 
(Möbius inversion in python, degree-rips using [persistable](https://github.com/LuisScoccola/persistable) and [RIVET](https://github.com/rivetTDA/rivet/)), [Mathieu Carrière](https://www-sop.inria.fr/members/Mathieu.Carriere/) (Sliced Wasserstein).

## Contributions
Hannah Schreiber

Feel free to contribute, report a bug on a pipeline, or ask for documentation by opening an issue.<br>
**Any** contribution is welcome.




## For mac users 
Due to the clang compiler, one may have to disable a compilator optimization to compile `multipers`: in the `setup.py` file, add the 
```bash
-fno-aligned-new
```
line in the `extra_compile_args` list. You should have should end up with something like the following.
```python
extensions = [Extension(f"multipers.{module}",
	sources=[f"multipers/{module}.pyx"],
	language='c++',
	extra_compile_args=[
		"-Ofast",
		"-std=c++20",
		"-fno-aligned-new",
		'-ltbb',
		"-Wall",
	],
	extra_link_args=['-ltbb'],
	define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
) for module in cython_modules]
```
#### Alternatives
One may try to use the `clang` compiler provided by conda or brew. If you have a simpler alternative, please let me know ;)
