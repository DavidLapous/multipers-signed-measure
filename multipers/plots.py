import matplotlib.pyplot as plt
import numpy as np

def _plot_rectangle(rectangle:np.ndarray, weight):
	x_axis=rectangle[[0,2]]
	y_axis=rectangle[[1,3]]
	color = "blue" if weight > 0 else "red"
	plt.plot(x_axis, y_axis, c=color)

def _plot_signed_measure_2(pts, weights):
	color_weights = np.empty(weights.shape, dtype=int)
	color_weights[weights>0] = 1
	color_weights[weights<0] = -1
	plt.scatter(pts[:,0],pts[:,1], c=color_weights, cmap="coolwarm")
def _plot_signed_measure_4(pts, weights):
	for rectangle, weight in zip(pts, weights):
		_plot_rectangle(rectangle=rectangle, weight=weight)

def plot_signed_measure(signed_measure, ax=None):
	if ax is None:
		ax = plt.gca()
	else:
		plt.sca(ax)
	pts, weights = signed_measure
	num_parameters = pts.shape[1]

	if isinstance(pts,np.ndarray):
		pass
	else:
		import torch
		if isinstance(pts,torch.Tensor):
			pts = pts.detach().numpy()
		else:
			raise Exception("Invalid measure type.")
	
	assert num_parameters in (2,4)
	if num_parameters == 2:
		_plot_signed_measure_2(pts=pts,weights=weights)
	else:
		_plot_signed_measure_4(pts=pts,weights=weights)

def plot_signed_measures(signed_measures, size=4):
	num_degrees = len(signed_measures)
	fig, axes = plt.subplots(nrows=1,ncols=num_degrees, figsize=(num_degrees*size,size))
	if num_degrees == 1:
		axes = [axes]
	for ax, signed_measure in zip(axes,signed_measures):
		plot_signed_measure(signed_measure=signed_measure,ax=ax)
	plt.tight_layout()


def plot_surface(grid, hf, fig=None, ax=None, **plt_args):
	if ax is None:
		ax = plt.gca()
	else:
		plt.sca(ax)
	# a,b = hf.shape
	# ax.set(xticks=grid[1], yticks=grid[0])
	# ax.imshow(hf.T, origin='lower', interpolation=None, **plt_args)
	# ax.set_box_aspect((a)
	im = ax.pcolormesh(grid[0], grid[1], hf.T, **plt_args)
	# ax.set_aspect("equal")
	# ax.axis('off')
	fig = plt.gcf() if fig is None else fig
	fig.colorbar(im,ax=ax)
def plot_surfaces(HF, size=4, **plt_args):
	grid, hf = HF
	assert hf.ndim == 3
	num_degrees = hf.shape[0]
	fig, axes = plt.subplots(nrows=1,ncols=num_degrees, figsize=(num_degrees*size,size))
	if num_degrees == 1:
		axes = [axes]
	for ax, hf_of_degree in zip(axes,hf):
		plot_surface(grid=grid,hf=hf_of_degree, fig=fig,ax=ax,**plt_args)
	plt.tight_layout()
