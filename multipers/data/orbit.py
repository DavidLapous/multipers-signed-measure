import numpy as np
from sklearn.preprocessing import LabelEncoder
def orbit(n:int=1000, r:float=1., x0=[]):
		point_list=[]
		if len(x0) != 2:
			x,y=np.random.uniform(size=2)
		else:
			x,y = x0
		point_list.append([x,y])
		for _ in range(n-1):
			x = (x + r*y*(1-y)) %1
			y = (y + r*x*(1-x)) %1
			point_list.append([x,y])
		return np.asarray(point_list, dtype=float)

def get_orbit5k(num_pts = 1000,  num_data=5000):
	rs = [2.5, 3.5, 4, 4.1, 4.3]
	labels = np.random.choice(rs, size=num_data, replace=True)
	X = [orbit(n=num_pts, r=r) for r in labels]
	labels = LabelEncoder().fit_transform(labels)
	return X, labels
