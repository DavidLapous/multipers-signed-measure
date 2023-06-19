from multipers.simplex_tree_multi import SimplexTreeMulti
from typing import Iterable
import numpy as np
# TODO : make a st python flag for coordinate_st, with grid resolution.
def rank_invariant2d(simplextree:SimplexTreeMulti, grid_shape:np.ndarray|list, degree:int)->np.ndarray: ...
def hilbert(simplextree:SimplexTreeMulti, grid_shape:np.ndarray|list, degree:int)->np.ndarray:...
def euler(simplextree:SimplexTreeMulti, grid_shape:np.ndarray|list, degree:int=None,inverse:bool =False, zero_pad:bool=False)->np.ndarray:...
def signed_measure(
	simplextree:SimplexTreeMulti,
	grid_shape:Iterable[int]|int|None=None, 
	degree:int|None=None, 
	zero_pad:bool=True, 
	grid_conversion:Iterable[Iterable[float]]|None=None,
	unsparse:bool=False,
	plot:bool=False,
	invariant:str|None=None)->tuple[np.ndarray, np.ndarray] | np.ndarray:
	"""
	Computes a discrete signed measure from various invariants.

	Parameters
	----------
	- simplextree : SimplexTreeMulti
		The multifiltered complex on which to compute the invariant.
		The filtrations values are assumed to be the coordinates in that filtration, i.e. integers
	- grid_conversion : Iterable[int]
		Reconverts the coordinate signed measure in that grid. 
	- grid_shape : Iterable[int] or Int or None:
		The coordinate grid shape. 
		If int, every parameter gets this resolution. 
		If None and grid_conversion is also None, the grid is infered from the filtration values.
	- degree : int|None
		If the invariant is hilbert or the rank invariant, the homological degree.
		If the invariant is euler, this parameter is ignored.
	- zero_pad=True : bool
		Zeros out the end of the grid, to enfore a 0-mass measure.
	- unsparse = False : bool
		Unsparse the output.
	- invariant = None : str
		The invariant to use to compute the signed measure. 
		Possible options : 'euler' or 'hilbert' or 'rank_invariant'
	- plot = False : bool
		Produce a plot for 2-parameter filtration.
	
	Output
	------

	Default
	- dirac locations : np.ndarray of shape (num_diracs x simplextree.num_parameters)
	- dirac weights : np.ndarray of shape (num_diracs)
	
	if unsparse is true, returns the unsparsified tensor. 
	"""
	...
