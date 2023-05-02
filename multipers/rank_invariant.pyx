from cython cimport numeric
from libcpp.vector cimport vector
from libc.stdint cimport intptr_t
from libcpp cimport bool
from libcpp cimport int
from libcpp.utility cimport pair

ctypedef vector[vector[vector[vector[int]]]] rank2
ctypedef vector[vector[int]] grid2D
ctypedef vector[vector[vector[int]]] grid3D
ctypedef vector[vector[vector[vector[int]]]] grid4D
ctypedef pair[vector[vector[int]],vector[int]] signed_measure_type
from itertools import product





cdef extern from "multi_parameter_rank_invariant/rank_invariant.h" namespace "Gudhi::rank_invariant":
	rank2 get_2drank_invariant(const intptr_t, const vector[int]&, const int) nogil
	grid2D get_2Dhilbert(const intptr_t, const vector[int]&, const int, bool) nogil except +
	signed_measure_type get_signed_measure(const intptr_t, const vector[int]&, int, int, bool) nogil except +
	grid3D get_3Dhilbert(const intptr_t, const vector[int]&, const int) nogil except +
	grid4D get_4Dhilbert(const intptr_t, const vector[int]&, const int) nogil except +
	grid2D get_euler2d(const intptr_t, const vector[int]&, bool, bool) nogil except +
	grid3D get_euler3d(const intptr_t, const vector[int]&, bool, bool) nogil except +
	grid4D get_euler4d(const intptr_t, const vector[int]&, bool, bool) nogil except +


from multipers.simplex_tree_multi import SimplexTreeMulti # Typing hack

cimport numpy as cnp
cnp.import_array()
import numpy as np

# TODO : make a st python flag for coordinate_st, with grid resolution.
def rank_invariant2d(simplextree:SimplexTreeMulti, grid_shape:np.ndarray|list, int degree):
	cdef intptr_t ptr = simplextree.thisptr
	cdef int c_degree = degree
	cdef vector[int] c_grid_shape = grid_shape
	cdef vector[vector[vector[vector[int]]]] out
	with nogil:
		out = get_2drank_invariant(ptr, c_grid_shape, c_degree)
	return np.array(out, dtype=float)

cdef _hilbert2d(simplextree:SimplexTreeMulti, grid_shape:np.ndarray|list, int degree, bool mobius_inversion):
	# assert simplextree.num_parameters == 2
	cdef intptr_t ptr = simplextree.thisptr
	cdef int c_degree = degree
	cdef vector[int] c_grid_shape = grid_shape
	cdef grid2D out
	with nogil:
		out = get_2Dhilbert(ptr, c_grid_shape, c_degree, mobius_inversion)
	return np.array(out, dtype=int)
# cdef _sm_2d(simplextree:SimplexTreeMulti, grid_shape:np.ndarray|list, int degree):
# 	# assert simplextree.num_parameters == 2
# 	cdef intptr_t ptr = simplextree.thisptr
# 	cdef int c_degree = degree
# 	cdef vector[int] c_grid_shape = grid_shape
# 	cdef signed_measure_type out
# 	with nogil:
# 		out = get_2D_SM(ptr, c_grid_shape, c_degree)
# 	return np.array(out.first, dtype=int), np.array(out.second, dtype=int)

cdef _hilbert3d(simplextree:SimplexTreeMulti, grid_shape:np.ndarray|list, int degree):
	# assert simplextree.num_parameters == 3
	cdef intptr_t ptr = simplextree.thisptr
	cdef int c_degree = degree
	cdef vector[int] c_grid_shape = grid_shape
	cdef grid3D out
	with nogil:
		out = get_3Dhilbert(ptr, c_grid_shape, c_degree)
	return np.array(out, dtype=int)

cdef _hilbert4d(simplextree:SimplexTreeMulti, grid_shape:np.ndarray|list, int degree):
	# assert simplextree.num_parameters == 3
	cdef intptr_t ptr = simplextree.thisptr
	cdef int c_degree = degree
	cdef vector[int] c_grid_shape = grid_shape
	cdef grid4D out
	with nogil:
		out = get_4Dhilbert(ptr, c_grid_shape, c_degree)
	return np.array(out, dtype=int)


def hilbert(simplextree:SimplexTreeMulti, grid_shape:np.ndarray|list, degree:int):
	assert len(grid_shape) >= simplextree.num_parameters, "Grid shape not valid"
	if simplextree.num_parameters == 2:
		out = _hilbert2d(simplextree, grid_shape, degree, False)
		return out
	if simplextree.num_parameters == 3:
		return _hilbert3d(simplextree, grid_shape, degree)
	if simplextree.num_parameters == 4:
		return _hilbert4d(simplextree, grid_shape, degree)
	raise Exception(f"Number of parameter has to be 2,3, or 4.")

def euler2d(simplextree:SimplexTreeMulti, grid_shape:np.ndarray|list, bool inverse=True, bool zero_pad=False):
	cdef intptr_t ptr = simplextree.thisptr
	cdef vector[int] c_grid_shape = grid_shape
	cdef vector[vector[int]] out
	with nogil:
		out = get_euler2d(ptr, c_grid_shape, inverse, zero_pad)
	return np.array(out, dtype=int)
def euler3d(simplextree:SimplexTreeMulti, grid_shape:np.ndarray|list, bool inverse=True, bool zero_pad=False):
	cdef intptr_t ptr = simplextree.thisptr
	cdef vector[int] c_grid_shape = grid_shape
	cdef grid3D out
	with nogil:
		out = get_euler3d(ptr, c_grid_shape, inverse, zero_pad)
	return np.array(out, dtype=int)
def euler4d(simplextree:SimplexTreeMulti, grid_shape:np.ndarray|list, bool inverse=True, bool zero_pad=False):
	cdef intptr_t ptr = simplextree.thisptr
	cdef vector[int] c_grid_shape = grid_shape
	cdef grid4D out
	with nogil:
		out = get_euler4d(ptr, c_grid_shape,inverse, zero_pad)
	return np.array(out, dtype=int)

def euler(simplextree:SimplexTreeMulti, grid_shape:np.ndarray|list, degree:int=None, bool inverse=False, bool zero_pad=False):
	if simplextree.num_parameters == 2:
		return euler2d(simplextree, grid_shape, inverse, zero_pad)
	if simplextree.num_parameters == 3:
		return euler3d(simplextree, grid_shape, inverse, zero_pad)
	if simplextree.num_parameters == 4:
		return euler4d(simplextree, grid_shape, inverse, zero_pad)
	raise Exception(f"Number of parameter has to be 2,3, or 4.")


def signed_measure(
	simplextree:SimplexTreeMulti,
	grid_shape:np.ndarray|list=None, 
	degree:int|None=None, 
	bool zero_pad=True, 
	grid_conversion=None,
	bool unsparse=False):
	if degree is None:
		invariant=2
		degree=0
	else:
		invariant=1
	assert grid_conversion is not None or grid_shape is not None
	if grid_shape is None:
		grid_shape = [len(f) for f in grid_conversion]
	cdef intptr_t ptr = simplextree.thisptr
	cdef vector[int] c_grid_shape = grid_shape
	cdef signed_measure_type out
	cdef int cinvariant =invariant
	cdef int cdegree = degree
	with nogil:
		out = get_signed_measure(ptr, c_grid_shape, cinvariant, cdegree, zero_pad)
	pts, weights = np.asarray(out.first, dtype=int), np.asarray(out.second, dtype=int)
	if len(pts) == 0:	
		pts=np.empty(shape=(0,simplextree.num_parameters), dtype=float)
		if not unsparse:
			return pts,weights
	if unsparse:
		from torch import sparse_coo_tensor
		return np.asarray(sparse_coo_tensor(indices=pts.T,values=weights, size=grid_shape, dtype=weights.dtype).to_dense(), dtype=weights.dtype)
	
	if grid_conversion is not None:
		coords = np.empty(shape=pts.shape, dtype=float)
		for i in range(coords.shape[1]):
			coords[:,i] = grid_conversion[i][pts[:,i]]
	else:
		coords = pts
	return coords, weights


# def trivial_rectangle(rectangle:np.ndarray, betti:np.ndarray):
# 	"""
# 	Checks if rectangle is trivial, i.e.,
# 		- if the betti is zero
# 		- if birth is not smaller than death
# 	"""
# 	num_parameters = rectangle.shape[0] // 2
# 	birth = np.asarray(rectangle[:num_parameters])
# 	death = np.asarray(rectangle[num_parameters:])
# 	r = betti[*rectangle]
# 	if r == 0:  return True
# 	if np.any(birth>death): return True
# 	return False

# def rectangle_to_betti(rectangle:np.ndarray, betti:np.ndarray):
# 		"""
# 		Splits birth, death, r and does the grid conversion if necessary.
# 		"""
# 		birth = np.array(rectangle[:num_parameters])
# 		death = np.asarray(rectangle[num_parameters:])
# 		r = betti[*rectangle]

# 		if not grid_conversion is None:
# 			birth = np.asarray([grid_conversion[i,bi] for i,bi in enumerate(birth)])
# 			death = np.asarray([grid_conversion[i,di] for i,di in enumerate(death)])
# 		return birth, death, r



# def tensor_to_rectangle(betti:np.ndarray, plot = False, grid_conversion=None):
# 	"""
# 	Turns a betti/rank tensor to its rank decomposition format.

# 	Parameters
# 	----------
# 	- betti rank tensor format
# 	- plot : if true, plots the rectangle decomposition
# 	- grid_conversion : if the grid is non regular, it can be given here to correct the rectangle endpoints

# 	Returns
# 	-------
# 	a list of weighted rectangles of the form [min coordinates, max coordinates, rectangle weight]
# 	"""
	
# 	num_parameters = betti.ndim // 2
# 	assert num_parameters == 2 or not plot
# 	grid_iterator = product(*[range(i) for i in betti.shape])
# 	rectangle_list = [rectangle_to_betti(rectangle=np.asarray(rectangle), betti=betti)
# 						for rectangle in range(grid_iterator)
# 						if not trivial_rectangle(rectangle, betti)]
# 	if plot:
# 		for b, d, r in rectangle_list: ## TODO Clean this 
# 			b1,b2 = b
# 			d1, d2 = d
# 			c = "blue" if r > 0  else "red"
# 			plt.plot([b1,d1], [b2,d2], c=c)
# 	return rectangle_list



# ##################################### LUIS' BETTI PY 



# # ctypedef np.npy_intp SIZE_t

# # cdef signed_betti(hilbert_function):
# # 	# number of dimensions
# # 	assert hilbert_function.dtype == np.int
# # 	cdef np.int n = hilbert_function.ndim
	
# # 	# pad with zeros at the end so np.roll does not roll over
# # 	cdef np.ndarray[np.int, ndim=n] hf_padded = np.pad(hilbert_function, [[0, 1]]*n, dtype=np.int)
# # 	# all relevant shifts (e.g., if n=2, (0,0), (0,1), (1,0), (1,1))
# # 	cdef np.ndarray[np.int, ndim=2] shifts = np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.int)
# # 	cdef np.ndarray[np.int, ndim=1] padded_shape = [hf_padded.shape[i] for i in range(hf_padded.ndim)]
# # 	cdef np.ndarray[np.int, ndim=n] bn = np.zeros(padded_shape, dtype=np.int)
# # 	cdef np.ndarray[np.int, ndim=1] c_range = list(range(n))
# # 	for i in range(shifts.shape[0]):
# # 		# bn += ((-1)**np.sum(shifts[i])) * np.roll(hf_padded, shifts[i], axis=range(n))
# # 		#((-1)**np.sum(shifts[i], dtype=bool))
# # 		# 1 -2*np.sum(shifts[i], dtype=bool)
# # 		bn +=  (1 -2*np.sum(shifts[i,:], dtype=bool)) *  np.roll(hf_padded, shifts[i,:], axis=c_range)

# # 	# with nogil:
# # 	# 	# for shift in shifts:
# # 	# 	# remove the padding
# # 	slices = np.ix_(*[range(0, hilbert_function.shape[i]) for i in range(n)])
# # 	return bn[slices]
	
