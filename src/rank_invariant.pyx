
ctypedef vector[vector[vector[vector[int]]]] rank2
ctypedef vector[vector[int]] grid2D

cdef extern from "rank_invariant/rank_invariant.h" namespace "Gudhi":
	rank2 get_2drank_invariant(const uintptr_t, const vector[unsigned int]&, const int) nogil
	grid2D get_2Dhilbert(const uintptr_t, const vector[unsigned int]&, const int) nogil

# TODO : make a st python flag for coordinate_st, with grid resolution.
def rank_inv(simplextree:MSimplexTree, grid_shape, degree:int):
	cdef uintptr_t ptr = simplextree.thisptr
	cdef int c_degree = degree
	cdef vector[unsigned int] c_grid_shape = grid_shape
	with nogil:
		out = get_2drank_invariant(ptr, c_grid_shape, c_degree)
	return np.asarray(out)

def hilbert2d(simplextree:MSimplexTree, grid_shape, degree:int):
	cdef uintptr_t ptr = simplextree.thisptr
	cdef int c_degree = degree
	cdef vector[unsigned int] c_grid_shape = grid_shape
	with nogil:
		out = get_2Dhilbert(ptr, c_grid_shape, c_degree)
	return np.asarray(out)
