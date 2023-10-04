
from typing import Callable, Iterable

import numpy as np
from joblib import Parallel, cpu_count, delayed, parallel_config
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

import multipers as mp


class SimplexTree2MMA(BaseEstimator, TransformerMixin):
	"""
	Turns a list of simplextrees to MMA approximations
	"""
	def __init__(self,n_jobs=-1, prune_degrees_above:int=None, progress=False, **persistence_kwargs) -> None:
		super().__init__()
		self.persistence_args = persistence_kwargs
		self.n_jobs=n_jobs
		self._has_axis=None
		self.prune_degrees_above=prune_degrees_above
		self.progress=progress
		return		
	def fit(self, X, y=None):
		self._has_axis = not isinstance(X[0], mp.SimplexTreeMulti)
		if self._has_axis:
			assert isinstance(X[0][0], mp.SimplexTreeMulti)
		return self
	def transform(self,X):
		if self.prune_degrees_above is not None:
			for x in X:
				x.prune_above_dimension(self.prune_degrees_above) # we only do for H0 for computational ease
		todo1 = lambda x : x.persistence_approximation(**self.persistence_args)
		if self._has_axis:
			todo = lambda sts : [todo1(st) for st in sts]
		else:
			todo=todo1
		return Parallel(n_jobs=self.n_jobs, prefer="threads")(delayed(todo)(x) for x in tqdm(X, desc="Computing modules", disable = not self.progress))


class MMAFormatter(BaseEstimator, TransformerMixin):

	def __init__(self, degrees:list=[0,1], axis=None, verbose:bool=False, normalize:bool=False,weights=None, quantiles=None):
		self._module_bounds=None
		self.verbose=verbose
		self.axis=None
		self._axis=None
		self._has_axis=None
		self._num_axis=None
		self.degrees=degrees
		self.normalize = normalize
		self._num_parameters = None
		self.weights = weights
		self.quantiles=quantiles
	@staticmethod
	def _complete_bound_with_box(x,degree):
		l,L = x.get_box()
		m,M = x.get_module_of_degree(degree).get_bounds()
		m = np.where(m<np.inf, m, l)
		M = np.where(M>-np.inf, M,L)
		return m,M
	
	@staticmethod
	def _infer_axis(X):
		has_axis = not isinstance(X[0], mp.PyModule)
		assert not has_axis or isinstance(X[0][0], mp.PyModule)
		return has_axis
	
	@staticmethod
	def _infer_num_parameters(X,ax=slice(None)):
		return X[0][ax].num_parameters
	
	@staticmethod 
	def _infer_bounds(X, degrees=None, axis=[slice(None)], quantiles=None):
		if degrees is None:
			degrees = np.arange(X[0][axis[0]].max_degree+1)
		bounds = np.array([[[MMAFormatter._complete_bound_with_box(x[ax],degree) for degree in degrees] for ax in axis] for x in X])
		if quantiles is not None:
			qm,qM = quantiles
			m = np.quantile(bounds[:,:,:,0,:], q=qm,axis=0)
			M = np.quantile(bounds[:,:,:,1,:], q=1-qM,axis=0)
		else:
			m = bounds[:,:,:,0,:].min(axis=0)
			M = bounds[:,:,:,1,:].max(axis=0)
		return (m,M)
	
	@staticmethod
	def _infer_grid(X:Iterable[mp.PyModule], strategy:str,resolution, degrees=None):
		num_parameters = X[0].num_parameters
		if degrees is None:
			filtration_values = tuple(mod.get_filtration_values(unique=True) for mod in X)
		else:
			filtration_values = tuple(mod.get_module_of_degrees(degrees).get_filtration_values(unique=True) for mod in X)
		
		if "_mean" in strategy:
			substrategy = strategy.split("_")[0]
			processed_filtration_values = [reduce_grid(f, resolution, substrategy, unique=False) for f in filtration_values]
			reduced_grid = np.mean(processed_filtration_values, axis=0)
		# elif "_quantile" in strategy:
		# 	substrategy = strategy.split("_")[0]
		# 	processed_filtration_values = [reduce_grid(f, resolution, substrategy, unique=False) for f in filtration_values]
		# 	reduced_grid = np.qu(processed_filtration_values, axis=0)
		else:
			filtration_values = [np.unique(np.concatenate([f[parameter] for f in filtration_values], axis=0)) for parameter in range(num_parameters)]
			reduced_grid = reduce_grid(filtration_values, resolution, strategy,unique=True)

		coordinates, new_resolution = filtration_grid_to_coordinates(reduced_grid, return_resolution=True)
		return coordinates,new_resolution
	
	def fit(self, X, y=None):
		if len(X) == 0: return self
		self._has_axis = self._infer_axis(X)
		# assert not self._has_axis or isinstance(X[0][0], mp.PyModule)
		if self.axis is None and self._has_axis:
			self.axis = -1
		if self.axis is not None and not (self._has_axis):
			raise Exception(f"SMF didn't find an axis, but requested axis {self.axis}")
		if self._has_axis:
			self._num_axis = len(X[0])
		if self.verbose:
			print('-----------MMAFormatter-----------')
			print('---- Infered stats')
			print(f'Found axis : {self._has_axis}, num : {self._num_axis}')
		
		self._axis = [slice(None)] if self.axis is None else range(self._num_axis) if self.axis == -1 else [self.axis]

		# X,axis,degree, min/max, parameter
		# bounds = np.array([[[self._complete_bound_with_box(x[ax],degree) for degree in self.degrees] for ax in self._axis] for x in X])
		# if self.quantiles is not None:
		# 	qm,qM = self.quantiles
		# 	m = np.quantile(bounds[:,:,:,0,:], q=qm,axis=0)
		# 	M = np.quantile(bounds[:,:,:,1,:], q=1-qM,axis=0)
		# else:
		# 	m = bounds[:,:,:,0,:].min(axis=0)
		# 	M = bounds[:,:,:,1,:].max(axis=0)
		# self._module_bounds = (m,M)
		self._module_bounds = self._infer_bounds(X,self.degrees, self._axis, self.quantiles)
		self._num_parameters = self._infer_num_parameters(X, ax=self._axis[0])
		assert self._num_parameters == self._module_bounds[0].shape[-1]
		if self.verbose:
			print(f'Number of parameters : {self._num_parameters}')
			print('---- Bounds :')
			if self._has_axis and self._num_axis>1:
				print('(axis) x (degree) x (parameter)')
			else:
				print('(degree) x (parameter)')
			print('-- Lower bound : ')
			m,M = self._module_bounds
			print(m.squeeze())
			print('-- Upper bound :')
			print(M.squeeze())

		w = 1 if self.weights is None else np.asarray(self.weights)
		m,M = self._module_bounds
		self._normalization_factors = w/(M-m)
		if self.verbose:
			print('-- Normalization factors:')
			print(self._normalization_factors.squeeze())

		if self.verbose:
			print('---- Module size :')
			for ax in self._axis:
				print(f'- Axis {ax}')
				for degree in self.degrees:
					sizes = [len(x[ax].get_module_of_degree(degree)) for x in X]
					print(f' - Degree {degree} size {np.mean(sizes).round(decimals=2)}±{np.std(sizes).round(decimals=2)}')
			print('----------------------------------')
		return self
	
	@staticmethod
	def copy_transform(mod, degrees, translation, rescale_factors, new_box):
		copy = mod.get_module_of_degrees(degrees) # and only returns the specific degrees
		for j,degree in enumerate(degrees): 
			copy.translate(translation[j], degree=degree)
			copy.rescale(rescale_factors[j], degree=degree)
		copy.set_box(new_box)
		return copy

	def transform(self, X):
		if self.normalize:
			if self.verbose: print("Normalizing...", end="")
			w = [1]*self._num_parameters if self.weights is None else np.asarray(self.weights)
			standard_box = mp.PyBox([0]*self._num_parameters, w)
			
			X_copy = [[
						self.copy_transform(
							mod=x[ax],
							degrees=self.degrees, 
							translation=-self._module_bounds[0][i],
							rescale_factors = self._normalization_factors[i], 
							new_box=standard_box)
				for i,ax in enumerate(self._axis)]
			for x in X]
			if self.verbose:	print("Done.")
		return X_copy
		# return [todo(x) for x in X]

class MMA2IMG(BaseEstimator, TransformerMixin):

	def __init__(self, 
			degrees:list, 
			bandwidth:float=0.1, 
			power:float=1, 
			normalize:bool=False, 
			resolution:list|int=50, 
			plot:bool=False, 
			box = None,
			n_jobs=1,
			flatten=False,
			progress=False,
			infer_grid_strategy="regular",
		):
		self.bandwidth=bandwidth
		self.degrees = degrees
		self.resolution=resolution
		self.box=box
		self.plot = plot 
		self._box=None
		self.normalize = normalize
		self.power = power
		self._has_axis=None
		self._num_parameters=None
		self.n_jobs=n_jobs
		self.flatten=flatten
		self.progress=progress
		self.infer_grid_strategy=infer_grid_strategy
		self._num_axis=None
	def fit(self, X, y=None):
		# TODO infer box
		# TODO rescale module
		self._has_axis = MMAFormatter._infer_axis(X)
		if self._has_axis:
			self._num_axis = len(X[0])
		if self.box is None:
			self._box = [[0],[1,1]]
		else: self._box = self.box 
		return self

	def transform(self, X):
		img_args = {
			"delta":self.bandwidth,
			"p":self.power,
			"normalize" : self.normalize,
			# "plot":self.plot,
			# "cb":1, # colorbar
			# "resolution" : self.resolution, # info in coordinates
			"box" : self.box,
			"degrees" : self.degrees,
		}
		if self._has_axis:
			its = (tuple(x[axis] for x in X) for axis in range(self._num_axis))
			crs = tuple(MMAFormatter._infer_grid(X_axis, self.infer_grid_strategy,self.resolution, degrees=self.degrees) for X_axis in its)
			coordss = [c for c,_ in crs]
			new_resolutions = [r for _, r in crs]
			todo1 = lambda x, c : x._compute_pixels(c, **img_args)
		else:
			coords, new_resolution = MMAFormatter._infer_grid(X, self.infer_grid_strategy,self.resolution, degrees=self.degrees)
			coordss = [coords]
			new_resolutions = [new_resolution]
			todo1 = lambda x : [x._compute_pixels(coords, **img_args)] # shape same as has_axis
		
		if self._has_axis:
			todo2 = lambda mods : [todo1(mod,c) for mod,c in zip(mods, coordss)]
		else:
			todo2 = todo1
		
		if self.flatten:
			todo = lambda x : np.concatenate(todo2(x),axis=0).flatten()
		else:
			todo = lambda flat_img_with_ax : [x.reshape(c,*r) for x in zip(flat_img_with_ax, coordss, new_resolutions)]

		# if self.flatten:
		# 	todo1 = lambda x, coords : x._compute_pixels(coords, degrees=self.degrees, flatten=self.flatten, **img_args)
		# else:
		# 	todo1 = lambda x, coords : x._compute_pixels(coords, degrees=self.degrees, flatten=self.flatten, **img_args).reshape((len(self.degrees,*new_resolution)))
		# if self._has_axis:
		# 	if self.flatten:
		# 		todo = lambda mods : np.concatenate([todo1(mod) for mod in mods],axis=0).flatten()
		# 	else:
		# 		todo = lambda mods : [todo1(mod) for mod in mods] # array not safe as they can have different resolutions
		# else:
		# 	todo = todo1
		return Parallel(n_jobs=self.n_jobs, prefer="threads")(delayed(todo)(x) for x in tqdm(X, desc="Computing images", disable = not self.progress)) ## res depends on ax (infer_grid)
		# return [todo(x) for x in X]






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
