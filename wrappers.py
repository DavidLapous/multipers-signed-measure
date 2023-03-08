
import numpy as np
import gudhi as gd
import multipers as mp
from tqdm import tqdm
from itertools import product
from sklearn.neighbors import KernelDensity
from sklearn.base import BaseEstimator, TransformerMixin
from warnings import warn
from signed_betti import *
from joblib import delayed
from types import FunctionType
from joblib import Parallel, delayed
from os.path import exists

from warnings import warn

def get_simplextree(x)->mp.SimplexTreeMulti:
	if isinstance(x, mp.SimplexTreeMulti):
		return x
	if len(x) == 3 and isinstance(x[0],FunctionType):
		f,args, kwargs = x
		return f(*args,**kwargs)
	else:
		warn("Not a valid SimplexTree !")
	return

################################################# Data2SimplexTree
class RipsDensity2SimplexTree(BaseEstimator, TransformerMixin):
	def __init__(self, bandwidth:float=1, threshold:float=np.inf, 
	sparse:float|None=None, num_collapse:int=0, max_dimension:int|None=None, 
	num_parameters:int=2, kernel:str="gaussian", delayed=False, rescale_density:float=0,
	progress:bool=False) -> None:
		super().__init__()
		self.bandwidth=bandwidth
		self.threshold = threshold
		self.sparse=sparse
		self.num_collapse=num_collapse
		self.max_dimension=max_dimension
		self.num_parameters = num_parameters
		self.kernel = kernel
		self.delayed=delayed
		self.rescale_density = rescale_density
		self.progress=progress
		return
	def fit(self, X:np.ndarray|list, y=None):
		if len(X) == 0:	return self
		if self.max_dimension is None:
			self.max_dimension = len(X[0])
		return self

	
	def transform(self,X):
		kde:KernelDensity=KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
		def get_st(x)->mp.SimplexTreeMulti:
			st= gd.RipsComplex(points = x, max_edge_length=self.threshold, sparse=self.sparse).create_simplex_tree(max_dimension=1)
			st=mp.SimplexTreeMulti(st, num_parameters = self.num_parameters)
			kde.fit(x)
			codensity = -kde.score_samples(x)
			if self.rescale_density != 0:
				codensity -= codensity.min()
				if codensity.max() != 0:	codensity /= codensity.max()
				codensity *= self.rescale_density
			st.fill_lowerstar(codensity, parameter = 1)
			st.collapse_edges(num=self.num_collapse)
			st.collapse_edges(num=self.num_collapse, strong = False, max_dimension = self.max_dimension)
			return st
		with tqdm(X, desc="Computing simplextrees", disable= not self.progress and self.delayed) as data:
			if self.delayed:
				return [delayed(get_st)(x) for x in data] # delay the computation for the to_module pipe
			return [get_st(x) for x in data] # No parallel possible here unless Gudhi serialize simplextrees.
			








################################################# SimplexTree2...

def _pts_convolution(weighted_pts:np.ndarray, filtration_grid, kernel="gaussian", bandwidth=0.1, **more_kde_args):
	kde = KernelDensity(kernel=kernel, bandwidth=bandwidth, **more_kde_args)
	grid_iterator = np.asarray(list(product(*filtration_grid)))
	grid_shape = [len(f) for f in filtration_grid]
	weights = weighted_pts[:,-1]
	pts = weighted_pts[:,:-1]
	pos_indices = weights>0
	neg_indices = weights<0
	img_pos = kde.fit(pts[pos_indices], sample_weight=weights[pos_indices]).score_samples(grid_iterator).reshape(grid_shape)
	img_neg = kde.fit(pts[neg_indices], sample_weight=-weights[neg_indices]).score_samples(grid_iterator).reshape(grid_shape)
	return np.exp(img_pos) - np.exp(img_neg)

def _st2ranktensor(st:mp.SimplexTreeMulti, filtration_grid:np.ndarray, degree:int, plot:bool, reconvert_grid:bool):
	"""
	TODO
	"""
	## Copy (the squeeze change the filtration values)
	stcpy = mp.SimplexTreeMulti(st)
	# turns the simplextree into a coordinate simplex tree
	stcpy.grid_squeeze(
		filtration_grid = filtration_grid, 
		coordinate_values = True)
	stcpy.collapse_edges(num=100, strong = True, ignore_warning=True)
	# computes the rank invariant tensor
	rank_tensor = mp.rank_inv(stcpy, degree=degree, grid_shape=[len(f) for f in filtration_grid])
	
	# refactor this tensor into the rectangle decomposition of the signed betti
	grid_conversion = filtration_grid if reconvert_grid else None 
	rectangle_decomposition = tensor_to_rectangle(
		betti=rank_decomposition_by_rectangles(rank_tensor), 
		plot=plot, grid_conversion=grid_conversion)
	return rectangle_decomposition

class SimplexTree2RectangleDecomposition(BaseEstimator,TransformerMixin):
	"""
	Transformer. 2 parameter SimplexTrees to their respective rectangle decomposition. 
	"""
	def __init__(self, filtration_grid:np.ndarray, degree:int, plot=False, reconvert_grid=True):
		super().__init__()
		self.filtration_grid = filtration_grid
		self.degree = degree
		self.plot=plot
		self.reconvert_grid = reconvert_grid
		return
	def fit(self, X, y=None):
		"""
		TODO : infer grid from multiple simplextrees
		"""
		return self
	def transform(self,X:list[mp.SimplexTreeMulti]):
		rectangle_decompositions = [
			_st2ranktensor(
				simplextree, filtration_grid=self.filtration_grid,
				degree=self.degree,
				plot=self.plot,
				reconvert_grid = self.reconvert_grid
			) 
			for simplextree in X
		]
		## TODO : return iterator ?
		return rectangle_decompositions

class SimplexTree2SignedMeasure(BaseEstimator,TransformerMixin):
	"""
	TODO
	"""
	def __init__(self, degrees:list[int],filtration_grid, grid_strategy="regular", bounds=None, progress=False, num_collapses=100, max_dimension=None, n_jobs=1):
		super().__init__()
		self.degrees = degrees
		self.grid_shape = [len(x) for x in filtration_grid]
		self.grid_strategy=grid_strategy
		self.filtration_grid = filtration_grid
		self.progress = progress
		self.bounds=bounds
		self.num_collapses=num_collapses
		self.max_dimension = max_dimension
		self.n_jobs = n_jobs
		return
	def fit(self, X, y=None):
		return self
	def transform1(self, simplextree):
		st = mp.SimplexTreeMulti(get_simplextree(simplextree)) ## COPY
		st.grid_squeeze(filtration_grid = self.filtration_grid, coordinate_values = True)
		if self.num_collapses == "full":
			st.collapse_edges(full=True,max_dimension=self.max_dimension)
		elif isinstance(self.num_collapses, int):
			st.collapse_edges(full=self.num_collapses,max_dimension=self.max_dimension)
		else:
			raise Exception("Bad edge collapse type. either 'full' or an int.")
		signed_measures = []
		for degree in self.degrees:
			rank = mp.hilbert2d(simplextree=st, degree=degree, grid_shape=self.grid_shape)
			betti = signed_betti(rank)
			signed_measure = betti_matrix2signed_measure(betti, grid_conversion=self.filtration_grid)
			if len(signed_measure) == 0:
				signed_measure = np.empty((0,st.num_parameters+1))
			signed_measures.append(signed_measure)
		return signed_measures
	def transform(self,X):
		return Parallel(n_jobs=self.n_jobs)(
			delayed(self.transform1)(to_st) for to_st in tqdm(X, disable = not self.progress, desc="Computing Hilbert function")
		)


class Hilbert2SignedMeasure(BaseEstimator,TransformerMixin):
	"""
	TODO
	"""
	def __init__(self, ):
		super().__init__()
		return
	def fit(self, X, y=None):
		return self
	def transform(self,X):
		return


class SignedMeasure2img(BaseEstimator,TransformerMixin):
	"""
	TODO
	"""
	def __init__(self, filtration_grid, kernel="gaussian", bandwidth=1., flatten:bool=False, **kwargs):
		super().__init__()
		self.kernel=kernel
		self.bandwidth=bandwidth
		self.more_kde_kwargs=kwargs
		self.filtration_grid=filtration_grid
		self.flatten=flatten
		return
	def fit(self, X, y=None):
		return self
	def transform(self,X):
		out =  [
			np.concatenate([
				_pts_convolution(
				signed_measure,self.filtration_grid, 
				kernel=self.kernel,
				bandwidth=self.bandwidth,
				**self.more_kde_kwargs
				)
			for signed_measure in signed_measures], axis=0) for signed_measures in X]
		if self.flatten:
			return [x.flatten() for x in out]
		return out


class SignedMeasure2SlicedWassersteinKernel(BaseEstimator,TransformerMixin):
	"""
	TODO
	"""
	def __init__(self):
		super().__init__()
		return
	def fit(self, X, y=None):
		return self
	def transform(self,X):
		return


def accuracy_to_csv(X,Y,cl, cln:str, k=10, dataset:str = "", filtration:str = "", shuffle=True,  verbose:bool=True, **kwargs):
	import pandas as pd
	if k>=1:
		from sklearn.model_selection import StratifiedKFold as KFold
		kfold = KFold(k, shuffle=shuffle).split(X,Y)
		accuracies = np.zeros(k)
		for i,(train_idx, test_idx) in enumerate(tqdm(kfold, total=k, desc="Computing kfold")):
			xtrain = [X[i] for i in train_idx]
			ytrain = [Y[i] for i in train_idx]
			cl.fit(xtrain, ytrain)
			xtest = [X[i] for i in test_idx]
			ytest = [Y[i] for i in test_idx] 
			accuracies[i] = cl.score(xtest, ytest)
			if verbose:	print(f"step {i}, {dataset} : {accuracies[i]}", flush=True)
	else:
		from sklearn.model_selection import train_test_split
		assert 0 < k < 1
		print("Computing accuracy, with train test split. Test size is defined by k", flush=True)
		xtrain, xtest, ytrain, ytest = train_test_split(X, Y, shuffle=shuffle, test_size=k)
		print("Fitting...", end="", flush=True)
		cl.fit(xtrain, ytrain)
		print("Computing score...", end="", flush=True)
		accuracies = cl.score(xtest, ytest)
		print("Done.")
		if verbose:	print(f"Accuracy {dataset} : {accuracies} ")

	file_path:str = f"result_{dataset}.csv".replace("/", "_").replace(".off", "")
	columns:list[str] = ["dataset", "mean", "std"]
	if exists(file_path):
		df:pd.DataFrame = pd.read_csv(file_path)
	else:
		df:pd.DataFrame = pd.DataFrame(columns= columns)
	more_names = []
	more_values = []
	for key, value in kwargs.items():
		more_names.append(key)
		more_values.append(value)
	new_line:pd.DataFrame = pd.DataFrame([[dataset, np.mean(accuracies), np.std(accuracies)]+more_values], columns = columns+more_names)
	df = pd.concat([df, new_line])
	df.to_csv(file_path, index=False)