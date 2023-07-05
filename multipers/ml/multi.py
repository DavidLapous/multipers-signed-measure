import numpy as np
import gudhi as gd
import multipers as mp
from tqdm import tqdm
from itertools import product
from sklearn.neighbors import KernelDensity
from sklearn.base import BaseEstimator, TransformerMixin
from types import FunctionType
from typing import Callable,Iterable
from joblib import Parallel, delayed, cpu_count
from torch import Tensor
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.ndimage import gaussian_filter
from .signed_betti import signed_betti, rank_decomposition_by_rectangles
from .invariants_with_persistable import hf_degree_rips
from .sliced_wasserstein import SlicedWassersteinDistance, WassersteinDistance
from .convolutions import convolution_signed_measures

def get_simplex_tree_from_delayed(x)->mp.SimplexTreeMulti:
	f,args, kwargs = x
	return f(*args,**kwargs)

def get_simplextree(x)->mp.SimplexTreeMulti:
	if isinstance(x, mp.SimplexTreeMulti):
		return x
	if len(x) == 3 and isinstance(x[0],FunctionType):
		return get_simplex_tree_from_delayed(x)
	else:
		raise TypeError("Not a valid SimplexTree !")


def infer_grid_from_points(pts:np.ndarray, num:int, strategy:str):
	if strategy =="regular":
		min = np.min(pts, axis=0)
		max = np.max(pts, axis=0)
		return np.linspace(min, max, num=num).T
	if strategy =="quantile":
		return np.quantile(pts, q=np.linspace(0,1,num), axis=0).T
	if strategy == "exact":
		F = [np.unique(pts[:,i]) for i in range(pts.shape[1])]
		F = [np.linspace(f.min(), f.max(), num=num) if len(f) > num else f for f in F] # fallback to regular if too large
		return F

	raise Exception(f"Grid strategy {strategy} not implemented")

def get_filtration_weights_grid(num_parameters:int=2, resolution:int|Iterable[int]=3,*, min:float=0, max:float=20, dtype=float, remove_homothetie:bool=True, weights=None):
	"""
	Provides a grid of weights, for filtration rescaling.
	 - num parameter : the dimension of the grid tensor
	 - resolution :  the size of each coordinate
	 - min : minimum weight
	 - max : maximum weight
	 - weights : custom weights (instead of linspace between min and max)
	 - dtype : the type of the grid values (useful for int weights)
	"""
	from itertools import product
	# if isinstance(resolution, int):
	try:
		float(resolution)
		resolution = [resolution]*num_parameters
	except:
		None
	if weights is None:	weights = [np.linspace(start=min,stop=max,num=r, dtype=dtype) for r in resolution]
	try:
		float(weights[0]) # same weights for each filtrations
		weights = [weights] * num_parameters
	except:
		None
	out = np.asarray(list(product(*weights)))
	if remove_homothetie:
		_, indices = np.unique([x / x.max() for x in out if x.max() != 0],axis=0, return_index=True)
		out = out[indices]
	return list(out)



################################################# Data2SimplexTree
class RipsDensity2SimplexTree(BaseEstimator, TransformerMixin):
	def __init__(self, bandwidth:float=-0.1, threshold:float=np.inf, 
			sparse:float|None=None, num_collapses:int=0, 
			kernel:str="gaussian", delayed=False,
			expand_dim:int=1,
			progress:bool=False, 
			n_jobs:int=-1, 
			rtol:float=1e-3, 
			atol=1e-4, 
			fit_fraction:float=1,
			verbose:bool=False,
		) -> None:
		"""
		The Rips + Density 1-critical 2-filtration.

		Parameters
		----------
		 - bandwidth : real : The kernel density estimation bandwidth. If negative, it replaced by abs(bandwidth)*(radius of the dataset)
		 - threshold : real,  max edge lenfth of the rips
		 - sparse : real, sparse rips (c.f. rips doc)
		 - num_collapse : int, Number of edge collapses applied to the simplextrees
		 - expand_dim : int, expand the rips complex to this dimension.
		 - kernel : the kernel used for density estimation. Available ones are the ones from scikit learn, e.g., "gaussian", "exponential", "tophat".
		 - delayed : bool If true will delay the computation to the next pipeline (for multiprocess computations).
		 - progress : bool, shows the calculus status
		 - n_jobs : number of processes
		 - fit_fraction : real, the fraction of data on which to fit
		 - verbose : bool, Shows more information if true.
		
		Output
		------
		A list of SimplexTreeMulti whose first parameter is a rips and the second is the codensity.
		"""
		super().__init__()
		self.bandwidth=bandwidth
		self.threshold = threshold
		self.sparse=sparse
		self.num_collapses=num_collapses
		self.kernel = kernel
		self.delayed=delayed
		self.progress=progress
		self._bandwidth=None
		self._threshold=None
		self.n_jobs = n_jobs
		self.rtol=rtol
		self.atol=atol
		self._scale=None
		self.fit_fraction=fit_fraction
		self.expand_dim=expand_dim
		self.verbose=verbose
		return
	def _get_distance_quantiles(self, X, qs):
		if len(qs) == 0: 
			self._scale = []
			return []
		if self.progress: print("Estimating scale...", flush=True, end="")
		indices = np.random.choice(len(X),min(len(X), int(self.fit_fraction*len(X))+1) ,replace=False)
		distances = np.asarray([distance_matrix(x,x)[np.triu_indices(len(x),k=1)].flatten() for x in (X[i] for i in indices)]).flatten()
		diameter = distances.max()
		if self.threshold > 0:	diameter = min(diameter, self.threshold)
		self._scale = diameter * np.asarray(qs) 
		if self.progress: print(f"Done. Chosen scales {qs} are {self._scale}", flush=True)
		return self._scale
	def _get_st(self,x, bandwidth=None)->mp.SimplexTreeMulti:
		bandwidth = self._bandwidth if bandwidth is None else bandwidth
		kde=KernelDensity(bandwidth=bandwidth, kernel=self.kernel, rtol=self.rtol, atol=self.atol)
		st = gd.RipsComplex(points = x, max_edge_length=self._threshold, sparse=self.sparse).create_simplex_tree(max_dimension=1)
		st = mp.SimplexTreeMulti(st, num_parameters = 2)
		codensity = -kde.fit(x).score_samples(x)
		st.fill_lowerstar(codensity, parameter = 1)
		if self.verbose: print("Num simplices :", st.num_simplices())
		if isinstance(self.num_collapses, int):
			st.collapse_edges(num=self.num_collapses)
			# if self.progress: print("Num simplices after collapse :", st.num_simplices())
			if self.verbose: print(", after collapse :", st.num_simplices(), end="")
		elif self.num_collapses == "full":
			# if self.verbose: print("Num simplices before collapse :", st.num_simplices(), end="")
			st.collapse_edges(full=True)
			if self.verbose: print(", after collapse :", st.num_simplices(), end="")
		if self.expand_dim > 1:
			st.expansion(self.expand_dim)
			if self.verbose: print(", after expansion :", st.num_simplices(), end="")
		if self.verbose: print("")
		return st
	def fit(self, X:np.ndarray|list, y=None):
		## default value 0.1 * diameter # TODO rescale density
		qs = [q for q in [-self.bandwidth, -self.threshold] if 0 <= q <= 1]
		self._get_distance_quantiles(X, qs=qs)
		self._bandwidth = self.bandwidth if self.bandwidth > 0 else self._scale[0]
		self._threshold = self.threshold if self.threshold > 0 else self._scale[-1]
		# self.bandwidth = "silverman" ## not good, as is can make bandwidth not constant
		return self

	
	def transform(self,X)->list[mp.SimplexTreeMulti]:
		with tqdm(X, desc="Computing simplextrees", disable = not self.progress or self.delayed) as data:
			if self.delayed:
				return [delayed(self._get_st)(x) for x in data] # delay the computation for the to_module pipe, as simplextrees are not pickle-able.
			return Parallel(n_jobs=self.n_jobs, prefer="threads")(delayed(self._get_st)(x) for x in data) # not picklable so prefer threads is necessary.

		
		
		
class RipsDensity2SimplexTrees(RipsDensity2SimplexTree):
	"""
	Same as the pipeline without the 's', but computes multiple bandwidths at once. 
	output shape :
	(data) x (bandwidth) x (simplextree)
	"""
	def __init__(self, bandwidths:Iterable[float]=-0.1, **rips_density_arguments) -> None:
		super().__init__(**rips_density_arguments)
		self.bandwidths=bandwidths
		self._bandwidths=None
		return
	def fit(self, X:np.ndarray|list, y=None):
		## default value 0.1 * diameter # TODO rescale density
		# if  np.any(np.array(self.bandwidths) < 0) or self.threshold < 0:
		# 	self._get_scale(X)
		# self._bandwidths = [- b * self._scale if b < 0 else b for b in self.bandwidths]
		# self._threshold = - self.threshold * self._scale if self.threshold < 0 else self.threshold
		
		qs = [q for q in [*-np.asarray(self.bandwidths), -self.threshold] if 0 <= q <= 1]
		self._get_distance_quantiles(X, qs=qs)
		self._bandwidths = np.asarray([b if b > 0 else s for b,s in zip(self.bandwidths, self._scale)])
		self._threshold = self.threshold if self.threshold > 0 else self._scale[-1]
		return self

	def _get_sts(self, x, bandwidths=None):
		bandwidths = self._bandwidths if bandwidths is None else bandwidths
		return [self._get_st(x, bandwidth=bandwidth) for bandwidth in bandwidths]
	def transform(self,X):
		with tqdm(X, desc="Computing simplextrees", disable= not self.progress and self.delayed) as data:
			if self.delayed:
				return [[delayed(self._get_st)(x, bandwidth=bandwidth) for bandwidth in self._bandwidths] for x in data] # delay the computation for the to_module pipe, as simplextrees are not pickle-able.
			return Parallel(n_jobs=self.n_jobs, prefer="threads")(delayed(self._get_sts)(x) for x in data) # not picklable so prefer threads is necessary.


		
		
class SimplexTreeEdgeCollapser(BaseEstimator, TransformerMixin):
	def __init__(self, num_collapses:int=0, full:bool=False, max_dimension:int|None=None, n_jobs:int=1) -> None:
		super().__init__()
		self.full=full
		self.num_collapses=num_collapses
		self.max_dimension=max_dimension
		self.n_jobs=n_jobs
		return
	def fit(self, X:np.ndarray|list, y=None):
		return self
	def transform(self,X):
		edges_list = Parallel(n_jobs=-1, prefer="threads")(delayed(mp.SimplextreeMulti.get_edge_list)(x) for x in X)
		collapsed_edge_lists = Parallel(n_jobs=self.n_jobs)(delayed(mp._collapse_edge_list)(edges, full=self.full, num=self.num_collapses) for edges in edges_list) ## 
		collapsed_simplextrees = Parallel(n_jobs=-1, prefer="threads")(delayed(mp.SimplexTreeMulti._reconstruct_from_edge_list)(collapsed_edge_lists, swap=True, expand_dim = self.max_dimension))
		return collapsed_simplextrees




class SimplexTree2MMA(BaseEstimator, TransformerMixin):
	"""
	Turns a list of simplextrees to MMA approximations
	"""
	def __init__(self,n_jobs=-1, **persistence_kwargs) -> None:
		super().__init__()
		self.persistence_args = persistence_kwargs
		self.n_jobs=n_jobs
		self._is_input_delayed=None
		return		
	def fit(self, X, y=None):
		self._is_input_delayed = not isinstance(X[0], mp.SimplexTreeMulti)
		return self
	def transform(self,X)->list[mp.PyModule]:
		if self._is_input_delayed:
			todo = lambda x : get_simplex_tree_from_delayed(x).persistence_approximation(**self.persistence_args)
		else:
			todo = lambda x : x.persistence_approximation(**self.persistence_args)
		return Parallel(n_jobs=self.n_jobs, prefer="threads")(delayed(todo)(x) for x in X)

class MMA2Landscape(BaseEstimator, TransformerMixin):
	"""
	Turns a list of MMA approximations into Landscapes vectorisations
	"""
	def __init__(self, resolution=[100,100], degrees:list[int]|None = [0,1], ks:Iterable[int]=range(5), phi:Callable = np.sum, box=None, plot:bool=False, n_jobs=-1, filtration_quantile:float=0.01) -> None:
		super().__init__()
		self.resolution:list[int]=resolution
		self.degrees = degrees
		self.ks=ks
		self.phi=phi # Has to have a axis=0 !
		self.box = box
		self.plot = plot
		self.n_jobs=n_jobs
		self.filtration_quantile = filtration_quantile
		return
	def fit(self, X, y=None):
		if len(X) <= 0:	return
		assert X[0].num_parameters == 2, f"Number of parameters {X[0].num_parameters} has to be 2."
		if self.box is None:
			_bottom = lambda mod : mod.get_bottom()
			_top = lambda mod : mod.get_top()
			m = np.quantile(Parallel(n_jobs=self.n_jobs, prefer="threads")(delayed(_bottom)(mod) for mod in X), q=self.filtration_quantile, axis=0)
			M = np.quantile(Parallel(n_jobs=self.n_jobs, prefer="threads")(delayed(_top)(mod) for mod in X), q=1-self.filtration_quantile, axis=0)
			self.box=[m,M]
		return self
	def transform(self,X)->list[np.ndarray]:
		if len(X) <= 0:	return
		todo = lambda mod : np.concatenate([
				self.phi(mod.landscapes(ks=self.ks, resolution = self.resolution, degree=degree, plot=self.plot), axis=0).flatten()
				for degree in self.degrees
			]).flatten()
		return Parallel(n_jobs=self.n_jobs, prefer="threads")(delayed(todo)(x) for x in X)

############################################### Data2Signedmeasure

def tensor_möbius_inversion(tensor:Tensor|np.ndarray, grid_conversion:Iterable[np.ndarray]|None = None, plot:bool=False, raw:bool=False, num_parameters:int|None=None):
	betti_sparse = Tensor(tensor.copy()).to_sparse() # Copy necessary in some cases :(
	num_indices, num_pts = betti_sparse.indices().shape
	num_parameters = num_indices if num_parameters is None else num_parameters
	if num_indices == num_parameters: # either hilbert or rank invariant
		rank_invariant = False
	elif 2*num_parameters == num_indices:
		rank_invariant = True
	else:
		raise TypeError(f"Unsupported betti shape. {num_indices} has to be either {num_parameters} or {2*num_parameters}.")
	points_filtration = np.asarray(betti_sparse.indices().T, dtype=int)
	weights = np.asarray(betti_sparse.values(), dtype=int)

	if grid_conversion is not None:
		coords = np.empty(shape=(num_pts,num_indices), dtype=float)
		for i in range(num_indices):
			coords[:,i] = grid_conversion[i%num_parameters][points_filtration[:,i]]
	else:
		coords = points_filtration
	if (not rank_invariant) and plot:
		plt.figure()
		color_weights = np.empty(weights.shape)
		color_weights[weights>0] = np.log10(weights[weights>0])+2
		color_weights[weights<0] = -np.log10(-weights[weights<0])-2
		plt.scatter(points_filtration[:,0],points_filtration[:,1], c=color_weights, cmap="coolwarm")
	if (not rank_invariant) or raw: return coords, weights
	def _is_trivial(rectangle:np.ndarray):
		birth=rectangle[:num_parameters]
		death=rectangle[num_parameters:]
		return np.all(birth<=death) # and not np.array_equal(birth,death)
	correct_indices = np.array([_is_trivial(rectangle) for rectangle in coords])
	if len(correct_indices) == 0:	return np.empty((0, num_indices)), np.empty((0))
	signed_measure = np.asarray(coords[correct_indices])
	weights = weights[correct_indices]
	if plot:
		assert signed_measure.shape[1] == 4 # plot only the rank decompo for the moment
		def _plot_rectangle(rectangle:np.ndarray, weight:float):
			x_axis=rectangle[[0,2]]
			y_axis=rectangle[[1,3]]
			color = "blue" if weight > 0 else "red"
			plt.plot(x_axis, y_axis, c=color)
		for rectangle, weight in zip(signed_measure, weights):
			_plot_rectangle(rectangle=rectangle, weight=weight)
	return signed_measure, weights


class DegreeRips2SignedMeasure(BaseEstimator, TransformerMixin):
	def __init__(self, degrees:Iterable[int], min_rips_value:float, 
	      max_rips_value,max_normalized_degree:float, min_normalized_degree:float, 
		  grid_granularity:int, progress:bool=False, n_jobs=1, sparse:bool=False, 
		  _möbius_inversion=True,
		  fit_fraction=1,
		  ) -> None:
		super().__init__()
		self.min_rips_value = min_rips_value
		self.max_rips_value = max_rips_value
		self.min_normalized_degree = min_normalized_degree
		self.max_normalized_degree = max_normalized_degree
		self._max_rips_value = None
		self.grid_granularity = grid_granularity
		self.progress=progress
		self.n_jobs = n_jobs
		self.degrees = degrees
		self.sparse=sparse
		self._möbius_inversion = _möbius_inversion
		self.fit_fraction=fit_fraction
		return
	def fit(self, X:np.ndarray|list, y=None):
		if self.max_rips_value < 0:
			print("Estimating scale...", flush=True, end="")
			indices = np.random.choice(len(X),min(len(X), int(self.fit_fraction*len(X))+1) ,replace=False)
			diameters =np.max([distance_matrix(x,x).max() for x in (X[i] for i in indices)])
			print(f"Done. {diameters}", flush=True)
		self._max_rips_value = - self.max_rips_value * diameters if self.max_rips_value < 0 else self.max_rips_value
		return self
	
	def _transform1(self, data:np.ndarray):
		_distance_matrix = distance_matrix(data, data)
		signed_measures = []
		rips_values, normalized_degree_values, hilbert_functions, minimal_presentations = hf_degree_rips(
			_distance_matrix,
			min_rips_value = self.min_rips_value,
			max_rips_value = self._max_rips_value,
			min_normalized_degree = self.min_normalized_degree,
			max_normalized_degree = self.max_normalized_degree,
			grid_granularity = self.grid_granularity,
			max_homological_dimension = np.max(self.degrees),
		)
		for degree in self.degrees:
			hilbert_function = hilbert_functions[degree]
			signed_measure = signed_betti(hilbert_function, threshold=True) if self._möbius_inversion else hilbert_function
			if self.sparse:
				signed_measure = tensor_möbius_inversion(
					tensor=signed_measure,num_parameters=2,
					grid_conversion=[rips_values, normalized_degree_values]
				)
			if not self._möbius_inversion: signed_measure = signed_measure.flatten()
			signed_measures.append(signed_measure)
		return signed_measures
	def transform(self,X):
		return Parallel(n_jobs=self.n_jobs)(delayed(self._transform1)(data) 
		for data in tqdm(X, desc=f"Computing DegreeRips, of degrees {self.degrees}, signed measures.", disable = not self.progress))






################################################# SimplexTree2...



def _st2ranktensor(st:mp.SimplexTreeMulti, filtration_grid:np.ndarray, degree:int, plot:bool, reconvert_grid:bool, num_collapse:int|str=0):
	"""
	TODO
	"""
	## Copy (the squeeze change the filtration values)
	stcpy = mp.SimplexTreeMulti(st)
	# turns the simplextree into a coordinate simplex tree
	stcpy.grid_squeeze(
		filtration_grid = filtration_grid, 
		coordinate_values = True)
	# stcpy.collapse_edges(num=100, strong = True, ignore_warning=True)
	if num_collapse == "full":
		stcpy.collapse_edges(full=True, ignore_warning=True, max_dimension=degree+1)
	elif isinstance(num_collapse, int):
		stcpy.collapse_edges(num=num_collapse,ignore_warning=True, max_dimension=degree+1)
	else:
		raise TypeError(f"Invalid num_collapse={num_collapse} type. Either full, or an integer.")
	# computes the rank invariant tensor
	rank_tensor = mp.rank_invariant2d(stcpy, degree=degree, grid_shape=[len(f) for f in filtration_grid])
	# refactor this tensor into the rectangle decomposition of the signed betti
	grid_conversion = filtration_grid if reconvert_grid else None 
	rank_decomposition = rank_decomposition_by_rectangles(
		rank_tensor, threshold=True,
		)
	rectangle_decomposition = tensor_möbius_inversion(tensor = rank_decomposition, grid_conversion = grid_conversion, plot=plot, num_parameters=st.num_parameters)
	return rectangle_decomposition

class SimplexTree2RectangleDecomposition(BaseEstimator,TransformerMixin):
	"""
	Transformer. 2 parameter SimplexTrees to their respective rectangle decomposition. 
	"""
	def __init__(self, filtration_grid:np.ndarray, degrees:Iterable[int], plot=False, reconvert_grid=True, num_collapses:int=0):
		super().__init__()
		self.filtration_grid = filtration_grid
		self.degrees = degrees
		self.plot=plot
		self.reconvert_grid = reconvert_grid
		self.num_collapses=num_collapses
		return
	def fit(self, X, y=None):
		"""
		TODO : infer grid from multiple simplextrees
		"""
		return self
	def transform(self,X:Iterable[mp.SimplexTreeMulti]):
		rectangle_decompositions = [
			[_st2ranktensor(
				simplextree, filtration_grid=self.filtration_grid,
				degree=degree,
				plot=self.plot,
				reconvert_grid = self.reconvert_grid,
				num_collapse=self.num_collapses
			) for degree in self.degrees]
			for simplextree in X
		]
		## TODO : return iterator ?
		return rectangle_decompositions



class SimplexTree2SignedMeasure(BaseEstimator,TransformerMixin):
	"""
	Input
	-----
	Iterable[SimplexTreeMulti]

	Output
	------
	Iterable[ list[signed_measure for degree] ]

	signed measure is either (points : (n x num_parameters) array, weights : (n) int array ) if sparse, else an integer matrix.

	Parameters
	----------
	 - degrees : list of degrees to compute. None correspond to the euler characteristic
	 - filtration grid : the grid on which to compute. If None, the fit will infer it from
	   - fit_fraction : the fraction of data to consider for the fit, seed is controlled by the seed parameter
	   - resolution : the resolution of this grid
	   - filtration_quantile : filtrations values quantile to ignore
	   - infer_filtration_strategy:str : 'regular' or 'quantile' or 'exact'
	   - normalize filtration : if sparse, will normalize all filtrations.
	 - expand : expands the simplextree to compute correctly the degree, for flag complexes
	 - invariant : the topological invariant to produce the signed measure. Choices are "hilbert" or "euler". Will add rank invariant later.
	 - num_collapse : Either an int or "full". Collapse the complex before doing computation.
	 - _möbius_inversion : if False, will not do the mobius inversion. output has to be a matrix then.
	 - enforce_null_mass : Returns a zero mass measure, by thresholding the module if True.
	"""
	def __init__(self, 
			degrees:list[int]|None=None, # homological degrees
			filtration_grid:Iterable[np.ndarray]|None=None, # filtration values to consider. Format : [ filtration values of Fi for Fi:filtration values of parameter i] 
			progress=False, # tqdm
			num_collapses:int|str=0, # edge collapses before computing 
			n_jobs=1, 
			resolution:Iterable[int]|int|None=100, # when filtration grid is not given, the resolution of the filtration grid to infer
			sparse=True, # sparse output
			plot:bool=False, 
			filtration_quantile:float=0., # quantile for inferring filtration grid
			_möbius_inversion:bool=True, # wether or not to do the möbius inversion (not recommended to touch)
			expand=True, # expand the simplextree befoe computing the homology
			normalize_filtrations:bool=False,
			# exact_computation:bool=False, # compute the exact signed measure.
			infer_filtration_strategy:str="exact",
			seed:int=0, # if fit_fraction is not 1, the seed sampling
			fit_fraction = 1,  # the fraction of the data on which to fit
			invariant="_", 
			out_resolution:Iterable[int]|int|None=None,
			# _true_exact:bool=False,
			enforce_null_mass:bool=True,
		  ):
		super().__init__()
		self.degrees = degrees
		self.filtration_grid = filtration_grid
		self.progress = progress
		self.num_collapses=num_collapses
		self.n_jobs = cpu_count() if n_jobs <= 0 else n_jobs
		self.resolution = resolution
		self.plot=plot
		self.sparse=sparse
		self.filtration_quantile=filtration_quantile
		self.normalize_filtrations = normalize_filtrations # Will only work for non sparse output. (discrete matrices cannot be "rescaled")
		self.infer_filtration_strategy = infer_filtration_strategy
		assert not self.normalize_filtrations or self.sparse, "Not able to normalize a matrix without losing information. Will not normalize."
		assert resolution is not None or filtration_grid is not None or infer_filtration_strategy == "exact"
		self.num_parameter = None
		self._is_input_delayed = None
		self._möbius_inversion = _möbius_inversion
		self._reconversion_grid = None
		self.expand = expand
		self._refit_grid = filtration_grid is None # will only refit the grid if filtration_grid has never been given.
		self.seed=seed
		self.fit_fraction = fit_fraction
		self.invariant = invariant
		self._transform_st = None
		self._to_simplex_tree = None
		self.out_resolution = out_resolution
		# self._true_exact=_true_exact
		self.enforce_null_mass = enforce_null_mass
		return	

	def _infer_filtration(self,X):
		if self.progress:	print(f"Inferring filtration grid from simplextrees, with strategy {self.infer_filtration_strategy}...", end="", flush=True)
		np.random.seed(self.seed)
		indices = np.random.choice(len(X), min(int(self.fit_fraction* len(X)) +1, len(X)), replace=False)
		prefer = "processes" if self._is_input_delayed else "threads"
		if self.infer_filtration_strategy == "regular":
			get_filtration_bounds = lambda x : self._to_simplex_tree(x).filtration_bounds(q=self.filtration_quantile)
			filtration_bounds =  Parallel(n_jobs=self.n_jobs, prefer=prefer)(delayed(get_filtration_bounds)(x) for x in (X[idx] for idx in indices))
			box = [np.min(filtration_bounds, axis=(0,1)), np.max(filtration_bounds, axis=(0,1))]
			diameter = np.max(box[1] - box[0])
			box = np.array([box[0] -0.1*diameter, box[1] + 0.1 * diameter])
			self.filtration_grid = [np.linspace(*np.asarray(box)[:,i], num=self.resolution[i]) for i in range(len(box[0]))]
			if self.progress:	print("Done.")
			return
		get_st_filtration = lambda x : self._to_simplex_tree(x).get_filtration_grid(grid_strategy="exact")
		filtrations =  Parallel(n_jobs=self.n_jobs, prefer=prefer)(delayed(get_st_filtration)(x) for x in (X[idx] for idx in indices))
		num_parameters = len(filtrations[0])

		if self.infer_filtration_strategy == "exact":
			# unique + sort
			filtrations = [np.unique(np.concatenate([x[i] for x in filtrations])) for i in range(num_parameters)]
			# If the numer of exact filtrations is too high, will replace the np.unique by a linspace
			if self.resolution is not None:
				for i,(f,r) in enumerate(zip(filtrations, self.resolution)):
					if len(f) > r and r > 0:
						m,M = f[0], f[-1]
						filtrations[i] = np.linspace(start=m, stop=M, num=r)
		elif self.infer_filtration_strategy == "quantile":
			filtrations = [np.unique(np.quantile(np.concatenate([x[i] for x in filtrations]), q=np.linspace(0,1,num=self.resolution[i]))) for i in range(num_parameters)]
		else:
			raise Exception(f"Strategy {self.infer_filtration_strategy} is not implemented. Available are regular, exact, quantile.")
		# Adds a last one, to take essensial summands into account (This is also prevents the zero pad from destroying information)
		for i,f in enumerate(filtrations):
			m,M = f[0], f[-1]
			filtrations[i] = np.unique(np.append(f, M + 0.1 * (M-m)))
		
		self.filtration_grid = filtrations
		if self.progress:	print("Done.")
		return

	def fit(self, X, y=None): # Todo : infer filtration grid ? quantiles ?
		assert self.invariant != "_" or self._möbius_inversion
		self._is_input_delayed = not isinstance(X[0], mp.SimplexTreeMulti)
		if self._is_input_delayed:
			self._to_simplex_tree = get_simplex_tree_from_delayed
		else:
			self._to_simplex_tree = lambda x : x
		if isinstance(self.resolution, int):
			self.resolution = [self.resolution]*self._to_simplex_tree(X[0]).num_parameters
		self.num_parameter = len(self.filtration_grid) if self.resolution is None else len(self.resolution)
		# if self.exact_computation: 
		# 	self._fit_exact(X)
		# elif self._refit_grid:
		# 	self._refit(X)
		if self._refit_grid:
			self._infer_filtration(X=X)
		if self.out_resolution is None:
			self.out_resolution = self.resolution
		elif isinstance(self.out_resolution, int):
			self.out_resolution = [self.out_resolution]*len(self.resolution)
		
		if self.normalize_filtrations:
			# self._reconversion_grid = [np.linspace(0,1, num=len(f), dtype=float) for f in self.filtration_grid] ## This will not work for non-regular grids...
			self._reconversion_grid = [f/np.std(f) for f in self.filtration_grid] # not the best, but better than some weird magic
		# elif not self.sparse: # It actually renormalizes the filtration !!  
		# 	self._reconversion_grid = [np.linspace(0,r, num=r, dtype=int) for r in self.out_resolution] 
		else:
			self._reconversion_grid = self.filtration_grid
		# else: 
		# 	self._reconversion_grid = [np.linspace(0,1, num=,) for _ in range]
		
		if self.invariant == "hilbert":
			def transform_hilbert(simplextree:mp.SimplexTreeMulti, degree:int, grid_shape:Iterable[int], _reconversion_grid):
				hilbert = mp.hilbert(simplextree=simplextree, degree=degree, grid_shape=grid_shape)
				signed_measure = signed_betti(hilbert, threshold=self.enforce_null_mass, sparse=False) if self._möbius_inversion else hilbert
				if self.sparse:
					signed_measure = tensor_möbius_inversion(tensor = signed_measure, 
					grid_conversion=_reconversion_grid, plot = self.plot, num_parameters=len(grid_shape))
				return signed_measure
			self._transform_st = transform_hilbert
		elif self.invariant == "euler":
			assert self.degrees == [None], f"Invariant euler incompatible with degrees {self.degrees}"
			def transform_euler(simplextree:mp.SimplexTreeMulti, degree:int, grid_shape:Iterable[int], _reconversion_grid):
				euler = mp.euler(simplextree=simplextree, degree=degree, grid_shape=grid_shape)
				signed_measure = signed_betti(euler, threshold=self.enforce_null_mass, sparse=False) if self._möbius_inversion else euler
				if self.sparse:
					signed_measure = tensor_möbius_inversion(tensor = signed_measure, 
					grid_conversion=_reconversion_grid, plot = self.plot, num_parameters=len(grid_shape))
				return signed_measure
			self._transform_st = transform_euler
			# self.degrees = [1000] # For the expansion
		elif self.invariant == "_":
			assert self._möbius_inversion is True
			def transform_sm(simplextree:mp.SimplexTreeMulti, degree:int|None, grid_shape:Iterable[int], _reconversion_grid):
				signed_measure = mp.signed_measure(
					simplextree=simplextree,degree=degree, 
					grid_shape=grid_shape, zero_pad=self.enforce_null_mass, 
					grid_conversion=_reconversion_grid, 
					unsparse = False,
					plot=self.plot,
					)

				if not self.sparse:
					# assert _reconversion_grid[0].dtype is int
					pts, weights = signed_measure
					bins = [[f.min(), f.max()] for f in _reconversion_grid]
					bins = [np.linspace(m-0.1*(M-m)/r, M+0.1*(M-m)/r, num=r+1) for (m,M),r in zip(bins, self.out_resolution)]
					signed_measure,_ = np.histogramdd(
						pts,bins=bins, 
						weights=weights
						)
					# print(signed_measure.shape)
				return signed_measure
			self._transform_st = transform_sm
		else:
			raise Exception(f"Bad invariant {self.invariant}. Pick either euler or hilbert.")
		return self
	def transform1(self, simplextree, filtration_grid=None, _reconversion_grid=None):
		if filtration_grid is None: filtration_grid = self.filtration_grid
		if _reconversion_grid is None: _reconversion_grid = self._reconversion_grid
		st = self._to_simplex_tree(simplextree)
		st = mp.SimplexTreeMulti(st, num_parameters = st.num_parameters) ## COPY
		st.grid_squeeze(filtration_grid = filtration_grid, coordinate_values = True)
		if st.num_parameters == 2:
			if self.num_collapses == "full":
				st.collapse_edges(full=True,max_dimension=1)
			elif isinstance(self.num_collapses, int):
				st.collapse_edges(num=self.num_collapses,max_dimension=1)
			else:
				raise Exception("Bad edge collapse type. either 'full' or an int.")
		signed_measures = []
		# print(st.num_simplices(),st.dimension(), self.degrees)
		if self.expand :
			max_degree = 1000 if self.degrees == [None] else np.max(self.degrees)+1
			st.expansion(max_degree)
		grid_shape = [len(f) for f in filtration_grid]
		for degree in self.degrees:
			signed_measure = self._transform_st(
				simplextree=st,degree=degree,
				grid_shape=grid_shape,
				_reconversion_grid=_reconversion_grid
			)
			signed_measures.append(signed_measure)
		return signed_measures
	def transform(self,X):
		assert self.filtration_grid is not None and self._transform_st is not None
		prefer = "processes" if self._is_input_delayed else "threads"
		out = Parallel(n_jobs=self.n_jobs, prefer=prefer)(
			delayed(self.transform1)(to_st) for to_st in tqdm(X, disable = not self.progress, desc=f"Computing topological invariant {self.invariant}")
		)
		return out
		# return [self.transform1(x) for x in tqdm(X, disable = not self.progress, desc="Computing Hilbert function")]





class SimplexTrees2SignedMeasures(SimplexTree2SignedMeasure):
	"""
	Input
	-----
	
	(data) x (axis, e.g. different bandwidths for simplextrees) x (simplextree)
	
	Output
	------ 
	(data) x (axis) x (degree) x (signed measure)
	"""
	def __init__(self,**kwargs):
		super().__init__(**kwargs)
		self._num_st_per_data=None
		# self._super_model=SimplexTree2SignedMeasure(**kwargs)
		self._filtration_grids = None
		return
	def fit(self, X, y=None):
		from sklearn.base import clone
		if len(X[0]) == 0: return self
		self._num_st_per_data = len(X[0])
		self._filtration_grids=[]
		for axis in range(self._num_st_per_data):
			self._filtration_grids.append(super().fit([x[axis] for x in X]).filtration_grid)
			# self._super_fits.append(truc)
		# self._super_fits_params = [super().fit([x[axis] for x in X]).get_params() for axis in range(self._num_st_per_data)]
		return self
	def transform(self, X):
		if self.normalize_filtrations:
			_reconversion_grids = [[np.linspace(0,1, num=len(f), dtype=float) for f in F] for F in self._filtration_grids]
		else:
			_reconversion_grids = self._filtration_grids
		def todo(x):
			# return [SimplexTree2SignedMeasure().set_params(**transformer_params).transform1(x[axis]) for axis,transformer_params in enumerate(self._super_fits_params)]
			return [
				self.transform1(x[axis],filtration_grid=filtration_grid, _reconversion_grid=_reconversion_grid) 
				for axis, filtration_grid, _reconversion_grid in zip(range(self._num_st_per_data), self._filtration_grids, _reconversion_grids)]
		return Parallel(n_jobs=-1, prefer="threads")(delayed(todo)(x) for x in X)


def rescale_sparse_signed_measure(signed_measure, filtration_weights, normalize_scales=None):
	from copy import deepcopy
	out = deepcopy(signed_measure)
	if normalize_scales is None:
		for degree in range(len(out)): # degree
			for parameter in range(len(filtration_weights)):
				out[degree][0][:,parameter] *= filtration_weights[parameter]
	else:
		for degree in range(len(out)):
			for parameter in range(len(filtration_weights)):
				out[degree][0][:,parameter] *= filtration_weights[parameter] / normalize_scales[degree][parameter]
	return out

class SignedMeasureFormatter(BaseEstimator,TransformerMixin):
	"""
	Input
	-----
	
	(data) x (degree) x (signed measure) or (data) x (axis) x (degree) x (signed measure)
	
	Iterable[list[signed_measure_matrix of degree]] or Iterable[previous].
	
	The second is meant to use multiple choices for signed measure input. An example of usage : they come from a Rips + Density with different bandwidth. 
	It is controlled by the axis parameter.

	Output
	------
	
	Iterable[list[(reweighted)_sparse_signed_measure of degree]]
	"""
	def __init__(self, 
			filtrations_weights:Iterable[float]=None,
			normalize=False,
			num_parameters:int|None=None,
			plot:bool=False,
			n_jobs:int=1, 
			unsparse:bool=False,
			axis:int=None,
			resolution:int|Iterable[int]=50,
			flatten:bool=False,
		):
		super().__init__()
		self.filtrations_weights = filtrations_weights
		self.num_parameters = num_parameters
		self.plot=plot
		self._grid =None
		self._old_shape = None
		self.n_jobs = n_jobs
		self.unsparse = unsparse
		self.axis=axis
		self._is_input_sparse=None
		self.resolution:int=resolution
		self._filtrations_bounds=None
		self.flatten=flatten
		self.normalize=normalize
		self._normalization_factors=None
		return
	def fit(self, X, y=None):
		## Gets a grid. This will be the max in each coord+1
		if len(X) == 0 or len(X[0]) == 0 or (self.axis is not None and len(X[0][0][0]) == 0):	return self
		
		self._is_input_sparse = (isinstance(X[0][0], tuple) and self.axis is None) or (isinstance(X[0][0][0], tuple) and self.axis is not None)
		# print("Sparse input : ", self._is_input_sparse)
		if self.axis is None:
			self.num_parameters = X[0][0][0].shape[1] if self._is_input_sparse else X[0][0].ndim
		else:
			#  (data) x (axis) x (degree) x (signed measure)
			self.num_parameters = X[0][0][0][0].shape[1] if self._is_input_sparse else X[0][0][0].ndim
		# Sets weights to 1 if None
		if self.filtrations_weights is None:
			self.filtrations_weights = np.array([1]*self.num_parameters)
		
		# resolution is iterable over the parameters
		try:
			float(self.resolution)
			self.resolution = [self.resolution]*self.num_parameters
		except:
			None
		assert len(self.filtrations_weights) == self.num_parameters == len(self.resolution), f"Number of parameter is not consistent. Inferred : {self.num_parameters}, Filtration weigths : {len(self.filtrations_weights)}, Resolutions : {len(self.resolution)}."
		# if not sparse : not recommended. 
		assert np.all(1 == np.asarray(self.filtrations_weights)) or self._is_input_sparse, f"Use sparse signed measure to rescale. Recieved weights {self.filtrations_weights}"

		if self.unsparse and self._is_input_sparse or self.normalize:
			if self.axis is None:
				stuff = [np.concatenate([sm[d][0] for sm in X], axis=0) for d in range(len(X[0]))]
				sizes_ = np.array([len(x)>0 for x in stuff])
				assert np.all(sizes_), f"Axis {not np.where(sizes_)} are trivial !"
				self._filtrations_bounds = np.asarray([[f.min(axis=0), f.max(axis=0)] for f in stuff])
			else:
				stuff = [np.concatenate([sm[self.axis][d][0] for sm in X], axis=0) for d in range(len(X[0][0]))]
				self._filtrations_bounds = np.asarray([[f.min(axis=0), f.max(axis=0)] for f in stuff])
			self._normalization_factors = self._filtrations_bounds[:,1] - self._filtrations_bounds[:,0] if self.normalize else None
			# print("Normalization factors : ",self._normalization_factors)
			if np.any(self._normalization_factors == 0 ):
				indices = np.where(self._normalization_factors == 0)
				# warn(f"Constant filtration encountered, at degree, parameter {indices} and axis {self.axis}.")
				self._normalization_factors[indices] = 1
		# assert self._is_input_sparse or not self.unsparse, "Cannot unsparse an already sparse matrix."
		
		# print(X[0])
		return self
	

	def unsparse_signed_measure(self, sparse_signed_measure:Iterable[tuple[np.ndarray, np.ndarray]]):
		filtrations = [np.linspace(start=a, stop=b, num=r) for (a,b),r in zip(self._filtrations_bounds, self.resolution)]
		# print(filtrations) #####
		out = []
		# print(sparse_signed_measure)
		for (pts, weights), filtration in zip(sparse_signed_measure, filtrations): # over degree
			signed_measure,_ = np.histogramdd(
				pts,bins=filtration.T, 
				weights=weights
				)
			if self.flatten:	signed_measure = signed_measure.flatten()
			out.append(signed_measure)
		if self.flatten:	out = np.concatenate(out).flatten()
		return out

	def transform(self,X):
		def todo_from_not_sparse(signed_measure:Iterable[np.ndarray]):
			if not self.flatten:
				return signed_measure
			return np.asarray([sm.flatten() for sm in signed_measure]).flatten()

		def todo_from_sparse(sparse_signed_measure:Iterable[tuple[np.ndarray, np.ndarray]]):
			out = rescale_sparse_signed_measure(sparse_signed_measure, filtration_weights=self.filtrations_weights, normalize_scales = self._normalization_factors)
			return out
			
		if self._is_input_sparse:
			todo = todo_from_sparse
		else:
			todo = todo_from_not_sparse
		
		if self.axis is None:
			it = X
		else:
			it = (x[self.axis] for x in X)
		out = Parallel(n_jobs=self.n_jobs, prefer="threads")(delayed(todo)(x) for x in it)

		if self.unsparse and self._is_input_sparse:
			# assert out[0][0][0].dtype is int, f"Can only unsparse coordinate values of signed measure ! Found {out[0][0][0].dtype}"
			out = [self.unsparse_signed_measure(x) for x in out]
			# print("Unsparse")
		# print(out[0][0].shape,np.abs(out[0][0]).max())
		return out










class SignedMeasure2Convolution(BaseEstimator,TransformerMixin):
	"""
	Discrete convolution of a signed measure

	Input
	-----
	
	(data) x (degree) x (signed measure)

	Parameters
	----------
	 - filtration_grid : Iterable[array] For each filtration, the filtration values on which to evaluate the grid
	 - resolution : int or (num_parameter) : If filtration grid is not given, will infer a grid, with this resolution
	 - infer_grid_strategy : the strategy to generate the grid. Available ones are regular, quantile, exact
	 - flatten : if true, the output will be flattened
	 - kernel : kernel to used to convolve the images.
	 - flatten : flatten the images if True
	 - progress : progress bar if True
	 - use_sklearn_convolution : Uses sklearn to compute convolutions, tends to be slower in this pipeline, but has more available kernels.
	 - plot : Creates a plot Figure.

	Output
	------
	
	(data) x (concatenation of imgs of degree)
	"""
	def __init__(self, 
	      filtration_grid:Iterable[np.ndarray]=None, 
		  kernel="gaussian", 
	      bandwidth:float|Iterable[float]=1., 
		  flatten:bool=False, n_jobs:int=1,
		  resolution:int|None=None, 
		  infer_grid_strategy:str="exact",
		  progress:bool=False, 
		  use_sklearn_convolution:bool=False,
		  plot:bool=False,
		  **kwargs):
		super().__init__()
		self.kernel=kernel
		self.bandwidth=bandwidth
		self.more_kde_kwargs=kwargs
		self.filtration_grid=filtration_grid
		self.flatten=flatten
		self.progress=progress
		self.n_jobs = n_jobs
		self.resolution = resolution
		self.infer_grid_strategy = infer_grid_strategy
		self._is_input_sparse = None
		self._refit = filtration_grid is None
		self._input_resolution=None
		self._bandwidths=None
		self.diameter=None
		self.use_sklearn_convolution=use_sklearn_convolution
		self.plot=plot
		return
	def fit(self, X, y=None):
		## Infers if the input is sparse given X 
		if len(X) == 0: return self
		if isinstance(X[0][0], tuple):	self._is_input_sparse = True 
		else: self._is_input_sparse = False
		# print(f"IMG output is set to {'sparse' if self.sparse else 'matrix'}")
		if not self._is_input_sparse:
			self._input_resolution = X[0][0].shape
			try:
				float(self.bandwidth)
				b = float(self.bandwidth)
				self._bandwidths = [b if b > 0 else -b * s for s in self._input_resolution]
			except:
				self._bandwidths = [b if b > 0 else -b * s for s,b in zip(self._input_resolution, self.bandwidth)]
			return self # in that case, singed measures are matrices, and the grid is already given
		
		if self.filtration_grid is None and self.resolution is None:
			raise Exception("Cannot infer filtration grid. Provide either a filtration grid or a resolution.")
		## If not sparse : a grid has to be defined
		if self._refit:
			# print("Fitting a grid...", end="")
			pts = np.concatenate([
				sm[0] for signed_measures in X for sm in signed_measures
			])
			self.filtration_grid = infer_grid_from_points(pts, strategy=self.infer_grid_strategy, num=self.resolution)
			# print('Done.')
		if self.filtration_grid is not None: self.diameter=np.linalg.norm([f.max() - f.min() for f in self.filtration_grid])
		return self
	
	def _sparsify(self,sm):
		return tensor_möbius_inversion(input=sm,grid_conversion=self.filtration_grid)

	def _sm2smi(self, signed_measures:Iterable[np.ndarray]):
			# print(self._input_resolution, self.bandwidths, _bandwidths)
		return np.concatenate([
				gaussian_filter(input=signed_measure, sigma=self._bandwidths,mode="constant", cval=0)
			for signed_measure in signed_measures], axis=0)
	# def _sm2smi_sparse(self, signed_measures:Iterable[tuple[np.ndarray]]):
	# 	return np.concatenate([
	# 			_pts_convolution_sparse(
	# 				pts = signed_measure_pts, pts_weights = signed_measure_weights,
	# 				filtration_grid = self.filtration_grid, 
	# 				kernel=self.kernel,
	# 				bandwidth=self.bandwidths,
	# 				**self.more_kde_kwargs
	# 			)
	# 		for signed_measure_pts, signed_measure_weights  in signed_measures], axis=0)
	def _transform_from_sparse(self,X):
		bandwidth = self.bandwidth if self.bandwidth > 0 else -self.bandwidth * self.diameter
		return convolution_signed_measures(X, filtrations=self.filtration_grid, bandwidth=bandwidth, flatten=self.flatten, n_jobs=self.n_jobs, kernel=self.kernel, sklearn_convolution=self.use_sklearn_convolution)
	
	def _plot_imgs(self, imgs:Iterable[np.ndarray]):
		extent = [self.filtration_grid[0][0], self.filtration_grid[0][-1], self.filtration_grid[1][0], self.filtration_grid[1][-1]]
		a,b,c,d = extent
		aspect =  (b-a) / (d-c) 
		num_degrees = imgs[0].shape[0]
		num_imgs = len(imgs)
		fig, axes = plt.subplots(nrows=num_degrees,ncols=num_imgs)
		if num_imgs==1:	axes=np.asarray([axes])
		if num_degrees == 1:	axes = np.asarray([axes])
		for j, img in enumerate(imgs):
			for i in range(num_degrees):
				plt.sca(axes[i,j])
				plt.imshow(img.T, origin="lower", extent=extent, cmap="Spectral", aspect=aspect)
		plt.show()
	def transform(self,X):
		if self._is_input_sparse is None:	raise Exception("Fit first")
		if self._is_input_sparse:
			out = self._transform_from_sparse(X)
		else:
			todo = SignedMeasure2Convolution._sm2smi
			out =  Parallel(n_jobs=self.n_jobs)(delayed(todo)(self, signed_measures) for signed_measures in tqdm(X, desc="Computing images", disable = not self.progress))
		if self.plot and not self.flatten:
			if self.progress:	print("Plotting convolutions...", end="")
			self._plot_imgs(out)
			if self.progress:	print("Done !")
		if self.flatten and not self._is_input_sparse:	out = [x.flatten() for x in out]

		return out



class SignedMeasure2SlicedWassersteinDistance(BaseEstimator,TransformerMixin):
	"""
	Transformer from signed measure to distance matrix.
	
	Input
	-----
	
	(data) x (degree) x (signed measure)

	Format
	------
	- a signed measure : tuple of array. (point position) : npts x (num_paramters) and weigths : npts
	- each data is a list of signed measure (for e.g. multiple degrees)

	Output
	------
	- (degree) x (distance matrix)
	"""
	def __init__(self, n_jobs:int=1, num_directions:int=10, _sliced:bool=True, epsilon=-1, ground_norm=1, progress = False, grid_reconversion=None, scales=None):
		super().__init__()
		self.n_jobs=n_jobs
		self._SWD_list = None
		self._sliced=_sliced
		self.epsilon = epsilon
		self.ground_norm = ground_norm
		self.num_directions = num_directions
		self.progress = progress
		self.grid_reconversion=grid_reconversion
		self.scales=scales
		return
		
	def fit(self, X, y=None):
		# _DISTANCE = lambda : SlicedWassersteinDistance(num_directions=self.num_directions) if self._sliced else WassersteinDistance(epsilon=self.epsilon, ground_norm=self.ground_norm) # WARNING if _sliced is false, this distance is not CNSD
		if len(X) == 0:	return self
		self.sparse = isinstance(X[0][0], tuple)
		num_degrees = len(X[0])
		self._SWD_list = [
			SlicedWassersteinDistance(num_directions=self.num_directions, n_jobs=self.n_jobs, scales=self.scales) 
			if self._sliced else 
			WassersteinDistance(epsilon=self.epsilon, ground_norm=self.ground_norm, n_jobs=self.n_jobs) 
			for _ in range(num_degrees)
		]
		for degree, swd in enumerate(self._SWD_list):
			signed_measures_of_degree = [x[degree] for x in X]
			if not self.sparse:	signed_measures_of_degree = [tensor_möbius_inversion(tensor=sm, grid_conversion=self.grid_reconversion) for sm in signed_measures_of_degree]
			swd.fit(signed_measures_of_degree)
		return self
	def transform(self,X):
		assert self._SWD_list is not None, "Fit first"
		out = []
		for degree, swd in tqdm(enumerate(self._SWD_list), desc="Computing distance matrices", total=len(self._SWD_list), disable= not self.progress):
			signed_measures_of_degree = [x[degree] for x in X]
			if not self.sparse:	signed_measures_of_degree = [tensor_möbius_inversion(tensor=sm, grid_conversion=self.grid_reconversion) for sm in signed_measures_of_degree]
			out.append(swd.transform(signed_measures_of_degree))
		return np.asarray(out)
	def predict(self, X): 
		return self.transform(X)


class SignedMeasures2SlicedWassersteinDistances(BaseEstimator,TransformerMixin):
	"""
	Transformer from signed measure to distance matrix.
	Input
	-----
	(data) x opt (axis) x (degree) x (signed measure)
	
	Format
	------
	- a signed measure : tuple of array. (point position) : npts x (num_paramters) and weigths : npts
	- each data is a list of signed measure (for e.g. multiple degrees)

	Output
	------
	- (axis) x (degree) x (distance matrix)
	"""
	def __init__(self, progress=False, n_jobs:int=1, scales:Iterable[Iterable[float]]|None = None, **kwargs): # same init
		self._init_child = SignedMeasure2SlicedWassersteinDistance(progress=False, scales=None,n_jobs=-1, **kwargs)
		self._axe_iterator=None
		self._childs_to_fit=None
		self.scales = scales
		self.progress = progress
		self.n_jobs=n_jobs
		return
		
	def fit(self, X, y=None):
		from sklearn.base import clone
		if len(X) == 0:	 return self
		if isinstance(X[0][0],tuple): # Meaning that there are no axes
			self._axe_iterator = [slice(None)]
		else:
			self._axe_iterator = range(len(X[0]))
		if self.scales is None: 
			self.scales = [None]
		else:
			self.scales = np.asarray(self.scales)
			if self.scales.ndim == 1:	
				self.scales = np.asarray([self.scales])
		assert self.scales[0] is None or self.scales.ndim==2, "Scales have to be either None or a list of scales !"
		self._childs_to_fit = [
			clone(self._init_child).set_params(scales=scales).fit(
				[x[axis] for x in X]) 
				for axis, scales in product(self._axe_iterator, self.scales)
			]
		print("New axes : ", list(product(self._axe_iterator, self.scales)))
		return self
	def transform(self,X):
		return Parallel(n_jobs=self.n_jobs//2 +1,)(
			delayed(self._childs_to_fit[child_id].transform)([x[axis] for x in X])
				for child_id, (axis, _) in tqdm(enumerate(product(self._axe_iterator, self.scales)), 
					desc=f"Computing distances matrices of axis, and scales", disable=not self.progress, total=len(self._childs_to_fit)
				) 
		)
		# [
		# 		child.transform([x[axis // len(self.scales)] for x in X]) 
		# 		for axis, child in tqdm(enumerate(self._childs_to_fit), 
		# 			desc=f"Computing distances of axis", disable=not self.progress, total=len(self._childs_to_fit)
		# 		)
		# 	]


