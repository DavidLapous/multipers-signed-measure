import multipers as mp
import numpy as np 
from scipy.sparse import coo_array
from scipy.ndimage import convolve1d
from multipers.ml import signed_betti







def test_1():
	st = mp.SimplexTreeMulti(num_parameters=3)
	st.insert([0], [1,0,0])
	st.insert([1], [0,1,0])
	st.insert([2], [0,0,1])
	st.insert([0,1,2], [2,2,2])
	st.make_filtration_non_decreasing()
	assert np.array_equal(mp.hilbert(st, grid_shape=[3,3,3], degree=0),np.array(
		[
	  [[0, 1, 1],
		[1, 2, 2],
		[1, 2, 2]],

		[[1, 2, 2],
		[2, 3, 3],
		[2, 3, 3]],

		[[1, 2, 2],
		[2, 3, 3],
		[2, 3, 1]]
		]
	))
	assert np.array_equal(mp.hilbert(st, grid_shape=[3,3,3], degree=0), mp.euler(st, grid_shape=[3,3,3], degree=0))

def test_2():
	st = mp.SimplexTreeMulti(num_parameters=4)
	st.insert([0], [1,0,0,0])
	st.insert([1], [0,1,0,0])
	st.insert([2], [0,0,1,0])
	st.insert([3], [0,0,0,1])
	st.insert([0,1,2,3], [2,2,2,2])
	st.make_filtration_non_decreasing()
	# list(st.get_simplices())
	assert np.array_equal((mp.hilbert(st, grid_shape=[3,3,3,3], degree=0)),(mp.euler(st, grid_shape=[3,3,3,3], degree=0)))

def test_3():
	st = mp.SimplexTreeMulti(num_parameters=2)
	st.insert([0,1,2], [1]*st.num_parameters)
	st.remove_maximal_simplex([0,1,2])
	a,b  = mp.signed_measure(st, degree=1, zero_pad=False)
	assert np.array_equal(a, [[1,1]]) and np.array_equal(b, [1])
	assert mp.signed_measure(st, degree=1)[1].sum() == 0

def test_4():
	st = mp.SimplexTreeMulti(num_parameters=3)
	st.insert([0], [1,0,0])
	st.insert([1], [0,1,0])
	st.insert([2], [0,0,1])
	st.insert([0,1,2], [2,2,2])
	st.make_filtration_non_decreasing()
	# list(st.get_simplices())
	assert np.array_equal(signed_betti(mp.euler(st, grid_shape=[3,3,3], degree=0)), mp.euler(st, grid_shape=[3,3,3], degree=0, inverse=True))
	assert np.array_equal(mp.signed_measure(st, grid_shape=[3,3,3], degree=0, unsparse=True), mp.euler(st, grid_shape=[3,3,3], degree=0, inverse=True, zero_pad=True))
	assert mp.signed_measure(st, degree=0)[1].sum() == 0


def test_5():
	st = mp.SimplexTreeMulti(num_parameters=4)
	st.insert([0], [1,0,0,0])
	st.insert([1], [0,1,0,0])
	st.insert([2], [0,0,1,0])
	st.insert([3], [0,0,0,1])
	st.insert([0,1,2,3], [2,2,2,2])
	st.make_filtration_non_decreasing()
	# list(st.get_simplices())
	assert np.array_equal(signed_betti(mp.hilbert(st, grid_shape=[3,3,3,3], degree=0)),mp.euler(st, grid_shape=[3,3,3,3], degree=0, inverse=True))
	assert np.array_equal(mp.signed_measure(st, grid_shape=[3,3,3,3], degree=0, unsparse=True), mp.euler(st, grid_shape=[3,3,3,3], degree=0, inverse=True, zero_pad=True))
	assert mp.signed_measure(st, grid_shape=[3,3,3,3], degree=0)[1].sum() == 0

def test_6():
	for num_parameters in range(2,4):
		st = mp.SimplexTreeMulti(num_parameters=num_parameters)
		f = np.random.randint(5,size=st.num_parameters)
		st.insert([0,1], f)
		st.insert([2,1], f)
		st.insert([0,3], f)
		st.insert([3,2], f)
		tensor = mp.signed_measure(st, degree=1, unsparse=True, zero_pad=False)
		tensor[*f]-=1
		assert np.all(tensor == 0), print(np.all(tensor==0))




