from typing import Iterable
from joblib import Parallel, delayed
import numpy as np
from itertools import product

from numba import njit, prange
import numba.np.unsafe.ndarray ## WORKAROUND FOR NUMBA

@njit(nogil=True,fastmath=True,inline="always", cache=True)
def _pts_convolution_gaussian_pt(pts, weights, pt, bandwidth):
	"""
	Evaluates the convolution of the signed measure (pts, weights) with a gaussian meaasure of bandwidth bandwidth, at point pt

	Parameters
	----------

	 - pts : (npts) x (num_parameters)
	 - weight : (npts)
	 - pt : (num_parameters)
	 - bandwidth : real

	Outputs
	-------

	The float value
	"""
	num_parameters = pts.shape[1]
	distances = np.empty(len(pts), dtype=float)
	for i in prange(len(pts)):
		distances[i] = np.sum((pt - pts[i])**2)/(2*bandwidth**2)
	distances = np.exp(-distances)*weights / (np.sqrt(2*np.pi)*(bandwidth**(num_parameters / 2))) # This last renormalization is not necessary
	return np.mean(distances)


@njit(nogil=True,fastmath=True,inline="always", cache=True)
def _pts_convolution_exponential_pt(pts, weights, pt, bandwidth):
	"""
	Evaluates the convolution of the signed measure (pts, weights) with a gaussian meaasure of bandwidth bandwidth, at point pt

	Parameters
	----------

	 - pts : (npts) x (num_parameters)
	 - weight : (npts)
	 - pt : (num_parameters)
	 - bandwidth : real

	Outputs
	-------

	The float value
	"""
	num_parameters = pts.shape[1]
	distances = np.empty(len(pts), dtype=float)
	for i in prange(len(pts)):
		distances[i] = np.linalg.norm(pt - pts[i])
	# distances = np.linalg.norm(pts-pt, axis=1)
	distances = np.exp(-distances/bandwidth)*weights / (bandwidth**num_parameters) # This last renormalization is not necessary
	return np.mean(distances)

@njit(nogil=True, inline="always", parallel=True, cache=True) # not sure if parallel here is worth it...
def _pts_convolution_sparse_pts(pts:np.ndarray, weights:np.ndarray, pt_list:np.ndarray, bandwidth, kernel:int=0):
	"""
	Evaluates the convolution of the signed measure (pts, weights) with a gaussian meaasure of bandwidth bandwidth, at points pt_list

	Parameters
	----------

	 - pts : (npts) x (num_parameters)
	 - weight : (npts)
	 - pt : (n)x(num_parameters)
	 - bandwidth : real

	Outputs
	-------

	The values : (n)
	"""
	if kernel == 0:
		return np.array([_pts_convolution_gaussian_pt(pts,weights,pt_list[i],bandwidth) for i in prange(pt_list.shape[0])])
	elif kernel == 1:
		return np.array([_pts_convolution_exponential_pt(pts,weights,pt_list[i],bandwidth) for i in prange(pt_list.shape[0])])
	else: 
		raise Exception("Unsupported kernel")

def convolution_signed_measures(iterable_of_signed_measures, filtrations, bandwidth, flatten:bool=True, n_jobs:int=1, sklearn_convolution=False, kernel="gaussian", **kwargs):
	"""
	Evaluates the convolution of the signed measures Iterable(pts, weights) with a gaussian measure of bandwidth bandwidth, on a grid given by the filtrations

	Parameters
	----------

	 - iterable_of_signed_measures : (num_signed_measure) x [ (npts) x (num_parameters), (npts)]
	 - filtrations : (num_parameter) x (filtration values)
	 - flatten : bool
	 - n_jobs : int

	Outputs
	-------

	The concatenated images, for each signed measure (num_signed_measures) x (len(f) for f in filtration_values)
	"""
	grid_iterator = np.array(list(product(*filtrations)), dtype=float)
	if sklearn_convolution:
		def convolution_signed_measures_on_grid(signed_measures:Iterable[tuple[np.ndarray,np.ndarray]]):
			return np.concatenate([
					_pts_convolution_sparse_old(pts=pts,pts_weights=weights, grid_iterator = grid_iterator, bandwidth= bandwidth, kernel=kernel, **kwargs) for pts,weights in signed_measures
				], axis=0)
	else:
		kernel2int = {"gaussian":0, "exponential":1, "other":2}
		def convolution_signed_measures_on_grid(signed_measures:Iterable[tuple[np.ndarray,np.ndarray]]):
			return np.concatenate([
					_pts_convolution_sparse_pts(pts,weights, grid_iterator, bandwidth, kernel=kernel2int[kernel]) for pts,weights in signed_measures
				], axis=0)

	
	if n_jobs>1 or n_jobs ==-1:
		convolutions = Parallel(n_jobs=-1, prefer="threads")(delayed(convolution_signed_measures_on_grid)(sms) for sms in iterable_of_signed_measures)
	else:	convolutions = [convolution_signed_measures_on_grid(sms) for sms in iterable_of_signed_measures]
	if not flatten:
		out_shape = [-1] + [len(f) for f in filtrations] # Degree
		convolutions = [x.reshape(out_shape) for x in convolutions]
	return np.asarray(convolutions, dtype=float)

def _test(r=1000, b=0.5, plot=True, kernel=0):
	import matplotlib.pyplot  as plt
	pts, weigths = np.array([[1.,1.], [1.1,1.1]]), np.array([1,-1])
	pt_list = np.array(list(product(*[np.linspace(0,2,r)]*2)))
	img = _pts_convolution_sparse_pts(pts,weigths, pt_list,b,kernel=kernel)
	if plot:
		plt.imshow(img.reshape(r,-1).T, origin="lower")
		plt.show()


def _pts_convolution_sparse_old(pts:np.ndarray, pts_weights:np.ndarray, grid_iterator, kernel="gaussian", bandwidth=0.1, **more_kde_args):
	"""
	Old version of `convolution_signed_measures`. Scikitlearn's convolution is slower than the code above.
	"""
	from sklearn.neighbors import KernelDensity
	if len(pts) == 0:
		# warn("Found a trivial signed measure !")
		return np.zeros(len(grid_iterator))
	kde = KernelDensity(kernel=kernel, bandwidth=bandwidth, rtol = 1e-4, **more_kde_args) # TODO : check rtol

	pos_indices = pts_weights>0
	neg_indices = pts_weights<0
	img_pos = kde.fit(pts[pos_indices], sample_weight=pts_weights[pos_indices]).score_samples(grid_iterator)
	img_neg = kde.fit(pts[neg_indices], sample_weight=-pts_weights[neg_indices]).score_samples(grid_iterator)
	return np.exp(img_pos) - np.exp(img_neg)




# def _pts_convolution_sparse(pts:np.ndarray, pts_weights:np.ndarray, filtration_grid:Iterable[np.ndarray], kernel="gaussian", bandwidth=0.1, **more_kde_args):
# 	"""
# 	Old version of `convolution_signed_measures`. Scikitlearn's convolution is slower than the code above.
# 	"""
# 	from sklearn.neighbors import KernelDensity
# 	grid_iterator = np.asarray(list(product(*filtration_grid)))
# 	grid_shape = [len(f) for f in filtration_grid]
# 	if len(pts) == 0:
# 		# warn("Found a trivial signed measure !")
# 		return np.zeros(shape=grid_shape)
# 	kde = KernelDensity(kernel=kernel, bandwidth=bandwidth, rtol = 1e-4, **more_kde_args) # TODO : check rtol
	
# 	pos_indices = pts_weights>0
# 	neg_indices = pts_weights<0
# 	img_pos = kde.fit(pts[pos_indices], sample_weight=pts_weights[pos_indices]).score_samples(grid_iterator).reshape(grid_shape)
# 	img_neg = kde.fit(pts[neg_indices], sample_weight=-pts_weights[neg_indices]).score_samples(grid_iterator).reshape(grid_shape)
# 	return np.exp(img_pos) - np.exp(img_neg)


### Precompiles the convolution
_test(r=2,b=.5, plot=False)
