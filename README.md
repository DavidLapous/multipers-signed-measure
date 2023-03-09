# multipers

## Installation
The following installs this `multipers` library
```sh
python setup.py build_ext --inplace
pip install -e .
```
The first line compiles the compiled `.so`s and the second links this python library to the current environment.
## Interface
This library has 3 modules
 - **simplex_tree_multi** : this is more or less the same as gudhi's simplex_tree, but with a vector filtration instead of a real filtration. There are a few usefull functions:
   - `mp.SimplexTreeMulti(<gudhi_simplex_tree>, num_parameters=2)` to convert a gudhi simplextree to a multiparameter simplextree
   - `SimplexTreeMulti.collapse_edges(<num_collapses>)`
   - `SimplexTreeMulti.grid_squeeze(<filtration_grid>, coordinate_values=False)` to project the filtration values on a grid
 - **rank_invariant** :
   - `mp.hilbert2d(<simplex_tree>, <degree>, <grid_shape>)` computes the 2-parameter hilbert function (output is a matrix of betti)
   - `mp.rank_invariant2d(<simplex_tree>, <degree>, <grid_shape>)` computes the 2-parameter rank invariant 
   output is : $$\mathrm{out}[i][j][k][l] = \mathrm{rk} M_{i,j} \to M_{k,l}$$
 - **multiparameter_module_approximation** : in order to visualize the $n$-modules, to debug.
   - Given a simplextree `st`, an approximation into interval decomposable modules can be computed using
   ```mod = st.persistence_approximation()```
   and if $n = 2$, `mod.plot(degree=1)` will produce a plot of $H_1$.
   - most of the other functions are explained on [this outdated notebook](https://github.com/DavidLapous/multipers/blob/main/How%20to%20use.ipynb)
   
   **Remark :** SimplexTreeMulti can encode multicritical filtration, but they are not handled yet by the 2 other modules.
