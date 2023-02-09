
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from signed_betti import rank_decomposition_by_rectangles, tensor_to_rectangle
import multipers



def _st2ranktensor(st:multipers.SimplexTree, filtration_grid:np.ndarray, degree:int, plot:bool, reconvert_grid:bool)->list[tuple[np.ndarray, np.ndarray, int]]:
	## Copy (the squeeze change the filtration values)
	stcpy = multipers.SimplexTree(st)
	# turns the simplextree into a coordinate simplex tree
	stcpy.grid_squeeze(
		filtration_grid = filtration_grid, 
		coordinate_values = True)
	stcpy.collapse_edges(num=100, strong = True, ignore_warning=True)
	# computes the rank invariant tensor
	rank_tensor = multipers.rank_inv(stcpy, degree=degree, grid_shape=[len(f) for f in filtration_grid])
	
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
	def transform(self,X:list[multipers.SimplexTree]):
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

class SimplexTree2Hilbert(BaseEstimator,TransformerMixin):
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

class Hilbert2SignedMeasure(BaseEstimator,TransformerMixin):
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

class SignedMeasure2img(BaseEstimator,TransformerMixin):
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
	
class RectangleDecomposition2SignedMeasure(BaseEstimator,TransformerMixin):
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
		
