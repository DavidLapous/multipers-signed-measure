from cython cimport numeric
from libcpp.vector cimport vector
from libc.stdint cimport intptr_t
from libcpp cimport bool
from libcpp cimport int
ctypedef vector[vector[vector[vector[int]]]] rank2
ctypedef vector[vector[int]] grid2D
from itertools import product





cdef extern from "multi_parameter_rank_invariant/rank_invariant.h" namespace "Gudhi::rank_invariant":
	rank2 get_2drank_invariant(const intptr_t, const vector[int]&, const int) nogil
	grid2D get_2Dhilbert(const intptr_t, const vector[int]&, const int) nogil

from multipers.simplex_tree_multi import SimplexTreeMulti # Typing hack

cimport numpy as cnp
cnp.import_array()
import numpy as np

# TODO : make a st python flag for coordinate_st, with grid resolution.
def rank_invariant2d(simplextree:SimplexTreeMulti, grid_shape:np.ndarray|list, int degree):
	cdef intptr_t ptr = simplextree.thisptr
	cdef int c_degree = degree
	cdef vector[int] c_grid_shape = grid_shape
#	cdef int I = grid_shape[0]
#	cdef int J = grid_shape[1]
#	cdef int[I][J][I][J] out_
#	with nogil:
#		out_ = get_2drank_invariant(ptr, c_grid_shape, c_degree)
#	cdef cnp.ndarray[int, ndim=4] out = out_
	return np.array(get_2drank_invariant(ptr, c_grid_shape, c_degree))

def hilbert2d(simplextree:SimplexTreeMulti, grid_shape:np.ndarray|list, int degree):
	cdef intptr_t ptr = simplextree.thisptr
	cdef int c_degree = degree
	cdef vector[int] c_grid_shape = grid_shape
	return np.array(get_2Dhilbert(ptr, c_grid_shape, c_degree))



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
	
