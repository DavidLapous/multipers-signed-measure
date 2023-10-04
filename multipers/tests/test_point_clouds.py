import multipers.ml.point_clouds as mmp
import numpy as np
import multipers as mp

def test_def():
	pts = np.array([[1,1],[2,2]], dtype=np.float32)
	st, = mmp.PointCloud2SimplexTree().fit_transform([pts])[0]
	assert isinstance(st, mp.simplex_tree_multi.SimplexTreeMulti)
	