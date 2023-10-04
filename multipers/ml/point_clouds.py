import numpy as np
import gudhi as gd
import multipers as mp
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Iterable
from multipers.ml.convolutions import KDE, DTM
from joblib import Parallel, delayed, parallel_config
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

# __all__=['RipsDensity2SimplexTree', 'RipsDensity2SimplexTrees', 'AlphaDTM2SimplexTrees', 'DTM']

class PointCloud2SimplexTree(BaseEstimator, TransformerMixin):
	def __init__(self, 
		bandwidth=.1,
		bandwidths=None, 
		threshold:float=np.inf,
		complex='rips',
		sparse:float|None=None, 
		num_collapses:int='full', 
		kernel:str="dtm", delayed=False,
		expand_dim:int=1,
		progress:bool=False, 
		n_jobs:int=-1, 
		fit_fraction:float=1,
		verbose:bool=False,
		) -> None:
		"""
		(Rips or Alpha) + (Density Estimation or DTM) 1-critical 2-filtration.

		Parameters
		----------
		 - bandwidth : real : The kernel density estimation bandwidth, or the DTM mass. If negative, it replaced by abs(bandwidth)*(radius of the dataset)
		 - threshold : real,  max edge lenfth of the rips or max alpha square of the alpha
		 - sparse : real, sparse rips (c.f. rips doc) WARNING : ONLY FOR RIPS
		 - num_collapse : int, Number of edge collapses applied to the simplextrees, WARNING : ONLY FOR RIPS
		 - expand_dim : int, expand the rips complex to this dimension. WARNING : ONLY FOR RIPS
		 - kernel : the kernel used for density estimation. Available ones are, e.g., "dtm", "gaussian", "exponential".
		 - progress : bool, shows the calculus status
		 - n_jobs : number of processes
		 - fit_fraction : real, the fraction of data on which to fit
		 - verbose : bool, Shows more information if true.
		
		Output
		------
		A list of SimplexTreeMulti whose first parameter is a rips and the second is the codensity.
		"""
		super().__init__()
		self.bandwidths=[bandwidth] if bandwidths is None else bandwidths
		self.threshold = threshold
		self.sparse=sparse
		self.num_collapses=num_collapses
		self.kernel = kernel
		self.delayed=delayed
		self.progress=progress
		self._bandwidths=None
		self._threshold=None
		self.n_jobs = n_jobs
		self._scale=None
		self.fit_fraction=fit_fraction
		self.expand_dim=expand_dim
		self.verbose=verbose
		self.complex=complex
		self._get_sts=None
		return
	def _get_distance_quantiles(self, X, qs):
		if len(qs) == 0: 
			self._scale = []
			return []
		if self.progress: print("Estimating scale...", flush=True, end="")
		indices = np.random.choice(len(X),min(len(X), int(self.fit_fraction*len(X))+1) ,replace=False)
		# diameter = np.asarray([distance_matrix(x,x).max() for x in (X[i] for i in indices)]).max()
		diameter = np.max([pairwise_distances(X = x, n_jobs = self.n_jobs).max() for x in (X[i] for i in indices)])
		self._scale = diameter * np.asarray(qs)
		if self.threshold > 0:	self._scale[self._scale>self.threshold] = self.threshold
		if self.progress: print(f"Done. Chosen scales {qs} are {self._scale}", flush=True)
		return self._scale
	

	def _get_rips_sts(self,x, codensities):
		# match input:
		# 	case 'points':
		# 		st = gd.RipsComplex(points = x, max_edge_length=self._threshold, sparse=self.sparse).create_simplex_tree(max_dimension=1)
		# 	case 'distance_matrix':
		# 		assert self.complex == 'rips' 
		# 		st = gd.RipsComplex(distance_matrix = x, max_edge_length=self._threshold, sparse=self.sparse).create_simplex_tree(max_dimension=1)
		# 	case _:
		# 		raise Exception('Invalid Rips imput. Either points or distance_matrix.')
		distance_matrix = pairwise_distances(X=x)
		def todo1(codensity):
			st = gd.RipsComplex(distance_matrix = distance_matrix, max_edge_length=self._threshold, sparse=self.sparse).create_simplex_tree(max_dimension=1)
			st = mp.simplex_tree_multi.SimplexTreeMulti(st, num_parameters = 2)
			st.fill_lowerstar(codensity, parameter = 1)
			if self.verbose: print("Num simplices :", st.num_simplices)
			if isinstance(self.num_collapses, int):
				st.collapse_edges(num=self.num_collapses)
				# if self.progress: print("Num simplices after collapse :", st.num_simplices)
				if self.verbose: print(", after collapse :", st.num_simplices, end="")
			elif self.num_collapses == "full":
				# if self.verbose: print("Num simplices before collapse :", st.num_simplices, end="")
				st.collapse_edges(full=True)
				if self.verbose: print(", after collapse :", st.num_simplices, end="")
			if self.expand_dim > 1:
				st.expansion(self.expand_dim)
				if self.verbose: print(", after expansion :", st.num_simplices, end="")
			if self.verbose: print("")
			return st
		
		return Parallel(prefer='threads')(delayed(todo1)(codensity) for codensity in codensities)
	
	def _get_alpha_sts(self,x, codensities, **unused_kwargs):
		st = gd.AlphaComplex(points=x).create_simplex_tree(max_alpha_square = self._threshold**2)
		st = mp.simplex_tree_multi.SimplexTreeMulti(st, num_parameters = 2)
		sts = [st]+[mp.simplex_tree_multi.SimplexTreeMulti(st, num_parameters = 2) for _ in codensities[1:]]
		for st, codensity in zip(sts, codensities): st.fill_lowerstar(codensity, parameter = 1)
		return sts

	# def _get_sts(self, **kwargs):
	# 	match self.complex:
	# 		case 'rips':
	# 			return self._get_rips_sts(**kwargs)
	# 		case 'alpha':
	# 			return self._get_alpha_sts(**kwargs)
	# 		case _:
	# 			raise ValueError("Invalid complex {_}")

	@staticmethod
	def _get_codensity_DTM(x,bandwidths):
		return DTM(masses=bandwidths,).fit(x).score_samples(x)

	@staticmethod
	def _get_codensity_KDE(x,bandwidths, kernel):
		return np.asarray([- KDE(bandwidth=bandwidth, kernel=kernel).fit(x).score_samples(x) for bandwidth in bandwidths])


	def _get_codensities(self,X, bandwidths):
		if self.kernel == 'dtm':
			return Parallel(n_jobs=self.n_jobs)(delayed(self._get_codensity_DTM)(x, bandwidths) for x in X)
		else:
			return Parallel(n_jobs=self.n_jobs)(delayed(self._get_codensity_KDE)(x, bandwidths, self.kernel) for x in X)
		
	
	def fit(self, X:np.ndarray|list, y=None):
		# self.bandwidth = "silverman" ## not good, as is can make bandwidth not constant
		match self.complex:
			case 'rips':
				self._get_sts = self._get_rips_sts
			case 'alpha':
				self._get_sts = self._get_alpha_sts
			case _:
				raise ValueError("Invalid complex {_}")
		
		qs = [q for q in [*-np.asarray(self.bandwidths), -self.threshold] if 0 <= q <= 1]
		self._get_distance_quantiles(X, qs=qs)
		self._bandwidths = np.array(self.bandwidths)
		count=0
		for i in range(len(self._bandwidths)):
			if self.bandwidths[i] < 0:
				self._bandwidths[i] = self._scale[count]
				count+=1
		self._threshold = self.threshold if self.threshold > 0 else self._scale[-1]
		
		return self

	
	def transform(self,X):
		with tqdm(X, desc="Computing simplextrees", disable = not self.progress or self.delayed) as data:
			codensitiess = self._get_codensities(X,self._bandwidths)
			if self.delayed:
				return [delayed(self._get_sts)(x) for x in data] # delay the computation for the to_module pipe, as simplextrees are not pickle-able.
			with parallel_config(n_jobs=self.n_jobs): 
				out = Parallel(prefer="threads")(delayed(self._get_sts)(x,codensities=codensities) for codensities,x in zip(codensitiess,X)) # not picklable so prefer threads is necessary.
			return out