import numpy as np
import itertools
from itertools import product
import matplotlib.pyplot as plt


def signed_betti(hilbert_function):
	# number of dimensions
	n = len(hilbert_function.shape)
	# pad with zeros at the end so np.roll does not roll over
	hf_padded = np.pad(hilbert_function, [[0, 1]]*n)
	# all relevant shifts (e.g., if n=2, (0,0), (0,1), (1,0), (1,1))
	shifts = np.array(list(itertools.product([0, 1], repeat=n)))
	bn = np.zeros(hf_padded.shape)
	for shift in shifts:
		bn += ((-1)**np.sum(shift)) * np.roll(hf_padded, shift, axis=range(n))
	# remove the padding
	slices = np.ix_(*[range(0, hilbert_function.shape[i]) for i in range(n)])
	return bn[slices]


def rank_decomposition_by_rectangles(rank_invariant):
	# takes as input the rank invariant of an n-parameter persistence module
	#   M  :  [0, ..., s_1 - 1] x ... x [0, ..., s_n - 1]  --->  Vec
	# on a grid with dimensions of sizes s_1, ..., s_n. The input is assumed to be
	# given as a tensor of dimensions (s_1, ..., s_n, s_1, ..., s_n), so that,
	# at index [i_1, ..., i_n, j_1, ..., j_n] we have the rank of the structure
	# map M(i) -> M(j), where i = (i_1, ..., i_n) and j = (j_1, ..., j_n), and
	# i <= j, meaning that i_1 <= j_1, ..., i_n <= j_n.
	# NOTE :
	#   - About the input, we assume that, if not( i <= j ), then at index
	#     [i_1, ..., i_n, j_1, ..., j_n] we have a zero.
	#   - Similarly, the output at index [i_1, ..., i_n, j_1, ..., j_n] only
	#     makes sense when i <= j. For indices where not( i <= j ) the output
	#     may take arbitrary values and they should be ignored.
	n = len(rank_invariant.shape)//2
	to_flip = tuple(range(n, 2 * n))
	return np.flip(signed_betti(np.flip(rank_invariant, to_flip)), to_flip)


def tensor_to_rectangle(betti:np.ndarray, plot = False, grid_conversion=None):
	"""
	Turns a betti/rank tensor to its rank decomposition format.

	Parameters
	----------
	- betti rank tensor format
	- plot : if true, plots the rectangle decomposition
	- grid_conversion : if the grid is non regular, it can be given here to correct the rectangle endpoints

	Returns
	-------
	a list of weighted rectangles of the form [min coordinates, max coordinates, rectangle weight]
	"""
	
	num_parameters = len(betti.shape) // 2
	assert num_parameters == 2 or not plot

	def trivial_rectangle(rectangle)-> bool:
		"""
		Checks if rectangle is trivial, i.e.,
			- if the betti is zero
			- if birth is not smaller than death
		"""
		birth = np.asarray(rectangle[:num_parameters])
		death = np.asarray(rectangle[num_parameters:])
		r = betti[*rectangle]
		if r == 0:  return True
		if np.any(birth>death): return True
		return False
	def rectangle_to_betti(rectangle:np.ndarray):
		"""
		Does the grid conversion if necessary.
		"""
		birth = np.asarray(rectangle[:num_parameters])
		death = np.asarray(rectangle[num_parameters:])
		r = betti[*rectangle]
		if not grid_conversion is None:
			birth = np.asarray([grid_conversion[i,bi] for i,bi in enumerate(birth)])
			death = np.asarray([grid_conversion[i,di] for i,di in enumerate(death)])
		return birth, death, r
	rectangle_list = [rectangle_to_betti(rectangle=np.asarray(rectangle))
						for rectangle in product(*[range(i) for i in betti.shape]) 
						if not trivial_rectangle(rectangle)]
	if plot:
		for b, d, r in rectangle_list: ## TODO Clean this 
			b1,b2 = b
			d1, d2 = d
			c = "blue" if r > 0  else "red"
			plt.plot([b1,d1], [b2,d2], c=c)
	return rectangle_list

def betti_matrix2signed_measure(betti, grid_conversion=None):
	if grid_conversion is  None:
		return np.asarray([
				[*indices, betti[*indices]]
				for indices in product(*[range(i) for i in betti.shape])
				if betti[*indices] != 0
			], dtype=int)
	return np.asarray([[*[grid_conversion[parameter][coord] for parameter, coord in enumerate(indices)], betti[*indices]]
			for indices in product(*[range(i) for i in betti.shape])
			if betti[*indices] != 0 ], dtype=float)


# only tests rank functions with 1 and 2 parameters
def test_rank_decomposition():

	# rank of an interval module in 1D on a grid with 2 elements
	ri = np.array([
		[
			1,  # 0,0
			1,  # 0,1
		],
		[
			0,  # 1,0
			1  # 1,1
		]
	]
	)
	expected_rd = np.array([
		[
			0,  # 0,0
			1,  # 0,1
		],
		[
			0,  # 1,0
			0  # 1,1
		]
	]
	)
	rd = rank_decomposition_by_rectangles(ri)
	for i in range(2):
		for i_ in range(i, 2):
			assert rd[i, i_] == expected_rd[i, i_]

	# rank of a sum of two rectangles in 2D on a grid of 2 elements
	ri = np.array([
		[[[1,  # (0,0), (0,0)
		 1],  # (0,0), (0,1)
		 [1,  # (0,0), (1,0)
		 1]  # (0,0), (1,1)
		  ],
		 [[0,  # (0,1), (0,0)
		   1],  # (0,1), (0,1)
		 [0,  # (0,1), (1,0)
		  1]  # (0,1), (1,1)
		  ]],
		[[[0,  # (1,0), (0,0)
		 0],  # (1,0), (0,1)
		 [2,  # (1,0), (1,0)
		 2]  # (1,0), (1,1)
		  ],
		 [[0,  # (1,1), (0,0)
		   0],  # (1,1), (0,1)
		 [0,  # (1,1), (1,0)
		  2]  # (1,1), (1,1)
		  ]]
	])
	expected_rd = np.array([
		[[[0,  # (0,0), (0,0)
		 0],  # (0,0), (0,1)
		 [0,  # (0,0), (1,0)
		 1]  # (0,0), (1,1)
		  ],
		 [[0,  # (0,1), (0,0)
		   0],  # (0,1), (0,1)
		 [0,  # (0,1), (1,0)
		  0]  # (0,1), (1,1)
		  ]],
		[[[0,  # (1,0), (0,0)
		 0],  # (1,0), (0,1)
		 [0,  # (1,0), (1,0)
		 1]  # (1,0), (1,1)
		  ],
		 [[0,  # (1,1), (0,0)
		   0],  # (1,1), (0,1)
		 [0,  # (1,1), (1,0)
		  0]  # (1,1), (1,1)
		  ]]
	])

	rd = rank_decomposition_by_rectangles(ri)
	for i in range(2):
		for i_ in range(i, 2):
			for j in range(2):
				for j_ in range(j, 2):
					assert rd[i,j,i_,j_] == expected_rd[i,j,i_,j_]

# only tests Hilbert functions with 1, 2, 3, and 4 parameters
def test_signed_betti():

	np.random.seed(0)
	N = 4

	# test 1D
	for _ in range(N):
		a = np.random.randint(10, 30)

		f = np.random.randint(0, 40, size=(a))
		sb = signed_betti(f)

		check = np.zeros(f.shape)
		for i in range(f.shape[0]):
			for i_ in range(0, i+1):
				check[i] += sb[i_]

		assert np.allclose(check, f)

	# test 2D
	for _ in range(N):
		a = np.random.randint(10, 30)
		b = np.random.randint(10, 30)

		f = np.random.randint(0, 40, size=(a, b))
		sb = signed_betti(f)

		check = np.zeros(f.shape)
		for i in range(f.shape[0]):
			for j in range(f.shape[1]):
				for i_ in range(0, i+1):
					for j_ in range(0, j+1):
						check[i, j] += sb[i_, j_]

		assert np.allclose(check, f)

	# test 3D
	for _ in range(N):
		a = np.random.randint(10, 20)
		b = np.random.randint(10, 20)
		c = np.random.randint(10, 20)

		f = np.random.randint(0, 40, size=(a, b, c))
		sb = signed_betti(f)

		check = np.zeros(f.shape)
		for i in range(f.shape[0]):
			for j in range(f.shape[1]):
				for k in range(f.shape[2]):
					for i_ in range(0, i+1):
						for j_ in range(0, j+1):
							for k_ in range(0, k+1):
								check[i, j, k] += sb[i_, j_, k_]

		assert np.allclose(check, f)

	# test 4D
	for _ in range(N):
		a = np.random.randint(5, 10)
		b = np.random.randint(5, 10)
		c = np.random.randint(5, 10)
		d = np.random.randint(5, 10)

		f = np.random.randint(0, 40, size=(a, b, c, d))
		sb = signed_betti(f)

		check = np.zeros(f.shape)
		for i in range(f.shape[0]):
			for j in range(f.shape[1]):
				for k in range(f.shape[2]):
					for l in range(f.shape[3]):
						for i_ in range(0, i+1):
							for j_ in range(0, j+1):
								for k_ in range(0, k+1):
									for l_ in range(0, l+1):
										check[i, j, k, l] += sb[i_, j_, k_, l_]

		assert np.allclose(check, f)
