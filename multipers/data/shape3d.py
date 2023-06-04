import numpy as np
from os.path import expanduser
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import FaceToEdge, NormalizeScale, SamplePoints
from torch_geometric.data.data import Data
import networkx as nx
DATASET_PATH = expanduser("~/Datasets/")
import os
from tqdm import tqdm


####################### MODELNET
def load_modelnet(version='10', sample_points = False, reset:bool=False):
	"""
	:param point_flag: Sample points if point_flag true. Otherwise load mesh
	:return: train_dataset, test_dataset
	"""
	assert version in ['10', '40']
	# if sample_points:
	# 	pre_transform, transform = FaceToEdge(remove_faces=False), SamplePoints(sample)
	# else:
	pre_transform, transform = FaceToEdge(remove_faces=True), None
	path = f"{DATASET_PATH}/ModelNet{version}"
	if reset:
		# print(f"rm -rf {path}")
		os.system(f"rm -rf {path+'/processed/'}")
	train_dataset = ModelNet(path, version, True, transform=transform, pre_transform=pre_transform)
	test_dataset = ModelNet(path, version, False, transform=transform, pre_transform=pre_transform)
	return train_dataset, test_dataset





def modelnet2graphs(version = '10', print_flag = False, labels_only = False, a = 0, b = 10, weight_flag = False):
	""" load modelnet 10 or 40 and convert to graphs"""
	train_dataset, test_dataset = load_modelnet(version, point_flag = False)
	dataset = train_dataset + test_dataset
	if b>0:	dataset = [dataset[i] for i in range(a,b)]
	if labels_only:
		return torch_geometric_2nx(dataset, labels_only=True)
	dataset = [FaceToEdge(remove_faces=False)(data) for data in dataset]
	graphs, labels = torch_geometric_2nx(dataset, print_flag=print_flag, weight_flag= weight_flag)
	return graphs, labels
def torch_geometric_2nx(dataset, labels_only = False, print_flag = False, weight_flag = False):
	"""
	:param dataset:
	:param labels_only: return labels only
	:param print_flag:
	:param weight_flag: whether computing distance as weights or not
	:return:
	"""
	if labels_only:
		return None, [int(data.y) for data in dataset]
	def data2graph(data:Data):
		edges = np.unique(data.edge_index.numpy().T, axis=0)
		g = nx.from_edgelist(edges)
		edge_filtration = {(u,v):np.linalg.norm(data.pos[u] - data.pos[v]) for u,v in g.edges}
		nx.set_node_attributes(g,{node:0 for node in g.nodes}, "geodesic")
		nx.set_edge_attributes(g, edge_filtration, "geodesic")
		return g
	return [data2graph(data) for data in tqdm(dataset, desc="Turning Data to graphs")], [int(data.y) for data in dataset]
def modelnet2pts2gs(version='10', nbr_size = 8, exp_flag = True, labels_only = False,n=100, n_jobs=1, random=False):
	""" sample points and create neighborhoold graph
	"""	
	train_dataset, test_dataset = load_modelnet(version=version, point_flag=True)
	dataset = train_dataset + test_dataset
	indices = np.random.choice(range(len(dataset)),replace=False, size=n) if random else range(n)

	dataset:list[Data] = [dataset[i] for i in indices]
	_,labels = torch_geometric_2nx(dataset, labels_only=True)
	if labels_only: return labels
	
	def data2graph(data:Data):
		pos = data.pos.numpy()
		adj = kneighbors_graph(pos, nbr_size, mode='distance', n_jobs=n_jobs) 
		g = nx.from_scipy_sparse_array(adj, edge_attribute= 'weight')
		if exp_flag:
			for u, v in g.edges(): # TODO optimize
				g[u][v]['weight'] = np.exp(-g[u][v]['weight'])
		return g
		#TODO : nx.set_edge_attributes()

	return [data2graph(data) for data in dataset], labels



def get_ModelNet(dataset, num_graph, seed):
	N = "" if num_graph <=0 else num_graph
	graphs_path = f"{DATASET_PATH}{dataset}/graphs{N}.pkl"
	labels_path = f"{DATASET_PATH}{dataset}/labels{N}.pkl"
	from os.path import exists
	if not exists(graphs_path) or not exists(labels_path):
		train,test = load_modelnet(version=dataset[8:])
		test_size = len(test) / len(train)
		if num_graph >0:
			np.random.seed(seed)
			indices = np.random.choice(len(train), N, replace=False)
			train = train[indices]
			indices = np.random.choice(len(test), int(N*test_size), replace=False)
			test = test[indices]
			np.random.seed() # resets seed
	

def get(dataset:str, num_graph=0, seed=0, node_per_graph=0):
	if dataset.startswith("ModelNet"):
		return get_ModelNet(dataset=dataset, num_graph=num_graph, seed=seed)
	datasets = get_(dataset=dataset, num_sample=num_graph)
	graphs = []
	labels = []
	np.random.seed(seed)
	for data, ls in datasets:
		nodes = np.random.choice(range(len(data.pos)), replace=False, size=node_per_graph)
		for i,node in enumerate(nodes):
			data_ = data # if i == 0 else None # prevents doing copies
			graphs.append([data_, node])
			labels.append(ls[node])
	return graphs, labels


def get_(dataset:str, dataset_num:int|None=None, num_sample:int=0, DATASET_PATH = expanduser("~/Datasets/")):
	from torch_geometric.io import read_off
	if dataset.startswith("3dshapes/"):
		dataset_ = dataset[len("3dshapes/"):]
	else:
		dataset_ = dataset
	if dataset_num is None and "/" in dataset_:
		position = dataset_.rfind("/")
		dataset_num = int(dataset_[position+1:-4]) # cuts the "<dataset>/" and the ".off"
		dataset_ = dataset_[:position]

	if dataset_num is None: # gets a random (available) number for this dataset
		from os import listdir
		from random import choice
		files = listdir(DATASET_PATH+f"3dshapes/{dataset_}")
		if num_sample <= 0:
			files = [file for file in files if "label" not in file]
		else:
			files = np.random.choice([file for file in files if "label" not in file], replace=False, size=num_sample)
		dataset_nums = np.sort([int("".join([char for  char in file  if char.isnumeric()])) for file in files])
		
		print("Dataset nums : ", *dataset_nums)
		out = [get_(dataset_, dataset_num=num) for num in dataset_nums]
		return out

	path = DATASET_PATH+f"3dshapes/{dataset_}/{dataset_num}.off"
	data = read_off(path)
	faces = data.face.numpy().T
	# data = FaceToEdge(remove_faces=remove_faces)(data)
	#labels 
	label_path = path.split(".")[0] + "_labels.txt"
	f = open(label_path, "r")
	labels = np.zeros(len(data.pos), dtype="<U10") # Assumes labels are of size at most 10 chars
	current_label=""
	for i, line in enumerate(f.readlines()):
		if i %  2 == 0:
			current_label = line.strip()
			continue
		faces_of_label = np.array(line.strip().split(" "), dtype=int) -1 # this starts at 1, python starts at 0
		# print(faces_of_label.min())
		nodes_of_label = np.unique(faces[faces_of_label].flatten())
		labels[nodes_of_label] = current_label  # les labels sont sur les faces
	return data, labels
