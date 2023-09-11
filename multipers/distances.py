import torch
import ot
import numpy as np


def sm2diff(sm1,sm2):
	if isinstance(sm1[0],np.ndarray):
		backend_concatenate = lambda a,b : np.concatenate([a,b], axis=0)
		backend_tensor = lambda x : np.asarray(x, dtype=int)
	elif isinstance(sm1[0],torch.Tensor):
		backend_concatenate = lambda a,b : torch.concatenate([a,b], dim=0)
		backend_tensor = lambda x :torch.tensor(x).type(torch.int)
	else:
		raise Exception("Invalid backend. Numpy or torch.")
	pts1,w1 = sm1
	pts2,w2 = sm2
	pos_indices1 = backend_tensor([i for i,w in enumerate(w1) for _ in range(w) if w>0])
	pos_indices2 = backend_tensor([i for i,w in enumerate(w2) for _ in range(w) if w>0]) 
	neg_indices1 = backend_tensor([i for i,w in enumerate(w1) for _ in range(-w) if w<0])
	neg_indices2 = backend_tensor([i for i,w in enumerate(w2) for _ in range(-w) if w<0])
	x = backend_concatenate(pts1[pos_indices1],pts2[neg_indices2])
	y = backend_concatenate(pts1[neg_indices1],pts2[pos_indices2])
	return x,y

def sm_distance(sm1,sm2, reg=0,reg_m=0, numItermax=10000, p=1):
	x,y = sm2diff(sm1,sm2)
	loss = ot.dist(x,y, metric='sqeuclidean', p=2) # only euc + sqeuclidian are implemented in pot for the moment with torch backend # TODO : check later
	empty_tensor = torch.tensor([]) # uniform weights
	if reg == 0:
		return ot.lp.emd2(empty_tensor,empty_tensor,M=loss)*len(x)
	if reg_m == 0:
		return ot.sinkhorn2(a=empty_tensor,b=empty_tensor,M=loss,reg=reg, numItermax=numItermax)
	return ot.sinkhorn_unbalanced2(a=empty_tensor,b=empty_tensor,M=loss,reg=reg, reg_m=reg_m, numItermax=numItermax)
	# return ot.sinkhorn2(a=onesx,b=onesy,M=loss,reg=reg, numItermax=numItermax)
	# return ot.bregman.empirical_sinkhorn2(x,y,reg=reg)