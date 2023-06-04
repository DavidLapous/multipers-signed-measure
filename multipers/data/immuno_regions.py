import numpy as np
from pandas import read_csv
from os.path import expanduser
from os import walk
from sklearn.preprocessing import LabelEncoder

def get(DATASET_PATH = expanduser("~/Datasets/")):
	DATASET_PATH += "1.5mmRegions/"
	X, labels = [],[]
	for label in ["FoxP3", "CD8", "CD68"]:
	#     for label in ["FoxP3", "CD8"]:
		for root, dirs, files in walk(DATASET_PATH + label+"/"):
			for name in files:
				X.append(np.array(read_csv(DATASET_PATH+label+"/"+name))/1500) ## Rescaled
				labels.append(label)
	return X, LabelEncoder().fit_transform(np.array(labels))
