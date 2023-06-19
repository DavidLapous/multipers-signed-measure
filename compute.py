from argparse import ArgumentParser, Namespace
import os
from joblib import cpu_count
import numpy as np

############################################ARGS PARSER
p = ArgumentParser()
p.add_argument("-d","--dataset", type=str, required=True, help="The dataset on which to do the computation. Either UCR, e.g., UCR/Coffee, graphs : graphs/BZR, orbit, 3dshapes, e.g., 3dshapes/Airplane ") #Threshold infinite values to compute diagram distance
p.add_argument("-p", "--pipeline", required=True,type=str, help="The pipeline to apply to the dataset. Available : dummy, filvec, pervec, sw, {rd,dr,multi}_{smi, smk, hilbert}, sw, smk, smh, smi, pl,pl_p, pi, pi_p. Where sm -> signed measure, sm{i,k,h} -> image, kernel, hilbert, rd -> rips+density bifiltration, dr-> degree+rips bifiltration, multi -> custom 1critical multi filtration (eg. graphs, molecules), sw -> sliced wasserstein, pl -> persistence landscape, pi -> persistance image.") # pipeline
p.add_argument("-fc","--final_classifier", default="xgboost", type=str, help="When the final input is a vector, this defines the final classifier") 
p.add_argument("-f","--filtration", default="", type=str, help="For 1 parameter filtration, the custom filtration. for example for graphs : ricciCurvature") # filtration on the graph (1-parameter)
p.add_argument("-fs","--filtrations", default=[], type=str, action="append", help = "For multifiltration, the filtrations to consider. e.g. --filtrations ricciCurvature --filtrations cc --filtrations geodesic. Depending on the dataset, available ones for graphs are cc,degree,fiedler,ricciCurvature,geodesic.") # filtrations on the graph (multi-parameter)
p.add_argument("-tk", "--train_k", default=10, type=int, help="Number of cross validations to choose the parameters during the training") # number of kfold for cross validation
p.add_argument("-k", "--test_k", default=10, type=float, help="Number of n-folds for testing. If 0<x<1, will do a train-test-split with a proportion of x for the test.") # number of kfold for test

p.add_argument("-t","--diagram_threshold", default=np.inf, type=float, help="For 1 parameter, thresholds persistence values to this threshold.") #Threshold infinite values to compute diagram distance
p.add_argument("-ns", "--num_samples", default=-1, type=int, help="number of data for orbit5k, and 3dshapes") # number of data (e.g. graph, or orbit data) samples
p.add_argument("-npts", "--num_pts", default=0, type=int, help="number of points / nodes in each data for 3dshapes / orbit") # number of pts per sample, if (synthetic)
p.add_argument("-res", "--resolution", type=int, default=100, help="For multiparameter pipelines, the resolution to compute the signed measure. e.g. 100 will do the computation on a [100]*num_parameter grid.")
p.add_argument("-ores", "--out_resolution", type=int, default=None, help="For multiparameter, vectorized pipelines, e.g. *_{smi, hilbert} the resolution of these vectors.")

p.add_argument("-is", "--infer_strategy", default="regular", help="Infers the grid on which to compute the topological invariant. Available : regular, quantile, exact.")
p.add_argument("-dq", "--drop_quantile", default=0.01, help="When inferring the filtrations, drop filtration values lower than this q and greater than 1-q.", type=float)
p.add_argument("-nr", "--num_rescales", default=1, help="Number of rescales per filtration for Kernel.", type=int)

p.add_argument("-rt", "--rips_threshold", type=float, default=np.inf, help="Maximum radius value for rips, when using a pipeline using rips.")

p.add_argument("-nf", "--normalize_filtrations", type=bool, default=True, help="Will rescale the (multi) filtrations by their standard deviation if true.")
# p.add_argument("-rb", "--rips_bandwidth", type=float, default=0., help="")
p.add_argument("-sr", "--sparse_rips", type=float, default=None, help="Value of the sparse rips, if using it.")
p.add_argument('-ext','--extended', action='append', type=int, default=[], help="Extended persistence for 1 parameter filtrations. if -1 : will use [0,2,5,7]. (Order given by gudhi)")
p.add_argument("-gb", "--geodesic_backend", default="torch_geometric")
p.add_argument("-s", "--seed", default=0, type=int, help="Some pipeline have randomized fit, this controls their seed.") # node selection seed
p.add_argument("-ti", "--topological_invariant", default="_", type=str, help="Available ones : _, hilbert, euler. _ is faster and will infer the invariant using the degree argument. If no degree is provided -> euler, else hilbert") # node selection seed
p.add_argument("-test", "--test", default=False, type=bool, help="Reduces the number of input, to ensure the pipelines are working. DO NOT USE WHEN NOT TESTING.") 

p.add_argument('-deg','--degrees', action='append', type=int, help="The homological degrees to considers. WARNING : this can also control the invariant used for multiparameter pipelines !")

p.add_argument("-j", "--n_jobs", default=cpu_count(), type=int, help="The number of threads to use.")
args = p.parse_args()


print("Loading dependencies...", end="", flush=True)
import multipers.ml.one as p1
import multipers.ml.multi as p2
import multipers.ml.kernels as mmk
import multipers.data.graphs as mdg
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from itertools import product
print("Done.", flush=True)



## FROM https://stackoverflow.com/questions/71845452/convert-argparse-namespace-to-dict-recursively
def namespace_to_dict(namespace):
	return {
		k: namespace_to_dict(v) if isinstance(v, Namespace) else v
		for k, v in vars(namespace).items()
	}
results_kwargs = namespace_to_dict(args) ## To be written in the end in the csv
dataset = results_kwargs.pop("dataset")


bandwidths = [-0.001, -0.01, -0.1, -0.2, -0.3]
# bandwidths = [-0.1]
results_kwargs["bandwidths"] = bandwidths




## ARGS magic
num_parameters = len(args.filtrations) if args.pipeline.startswith("multi_") else 2
args.grid_shape= [args.resolution]*num_parameters
shuffle = True if args.filtration != "dijkstra" else False
extended = args.extended
if len(extended) == 1 and extended[0] == -1:
	extended = [0,2,5,7] # ord0, ext+0, rel1, Ext-1
	degrees = list(range((max(extended) // 4)+1))
elif len(extended) > 0:
	extended = extended[1:]
	degrees = list(range((max(extended) // 4)+1))
else:
	degrees = args.degrees
	extended = False
args.extended = extended
args.degrees = degrees
# max_degree = np.max(args.degrees) +1
if args.out_resolution is None:	args.out_resolution = args.resolution

### Final classifiers
if args.final_classifier == "rf":
	final_classifier = RandomForestClassifier()
	final_classifier_parameters={}
elif args.final_classifier == "xgboost":
	final_classifier = XGBClassifier()
	final_classifier_parameters={}
elif args.final_classifier == "mlp":
	final_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
	final_classifier_parameters={}
	# TODO : CNN
elif args.final_classifier == "svm":
	final_classifier = SVC(kernel="rbf")
	final_classifier_parameters={}
else:
	raise Exception(f"Classifier {args.final_classifier} not implemented.")



###########################################DATASET / FILTRATIONS
print("Getting dataset", flush=True)
if args.dataset == "orbit":
	from multipers.data import orbit
	X,Y = orbit.get_orbit5k(num_data=args.num_samples, num_pts=args.num_pts)
elif args.dataset.startswith("UCR/"):
	from multipers.data import UCR
	xtrain, ytrain = UCR.get_train(dataset=args.dataset)
	xtest, ytest = UCR.get_test(dataset = args.dataset)
	## FOR ACCURACY 2 CSV
	args.test_k = len(xtest) / (len(xtrain) + len(xtest))
	shuffle = False
	X = xtrain + xtest
	Y = np.concatenate([ytrain, ytest])
	## TESTS THAT ACCURACY TO CSV RETRIEVES THE SAME
	from sklearn.model_selection import train_test_split
	_xtrain, _xtest, _ytrain, _ytest = train_test_split(X, Y, shuffle=shuffle, test_size=args.test_k)
	assert np.array_equal(xtrain, _xtrain)
	assert np.array_equal(ytrain,_ytrain)
	assert np.array_equal(xtest, _xtest)
	assert np.array_equal(ytest,_ytest)
elif args.dataset == "immuno":
	from multipers.data import immuno_regions
	X,Y = immuno_regions.get()
elif args.dataset.startswith("3dshapes/"):
	from multipers.data import shape3d
	X,Y = shape3d.get(dataset = args.dataset, num_graph=args.num_samples, node_per_graph = args.num_pts)
elif args.dataset.startswith("graphs/"):
	from multipers.data import graphs
	print("Checking graphs filtrations ...", flush=1)
	filtrations = args.filtrations
	if args.filtrations == ["reset"]:
		print("Computing all filtrations and leaving...", end="\n")
		graphs.compute_filtration(args.dataset, "ALL")
		print("Done")
		exit()
	if len(filtrations) == 0:
		assert args.filtration != "", "Provide a filtration for graph data!"
		filtrations = [args.filtration]
	for f in filtrations:
		graphs.get(dataset = args.dataset, filtration=f) # Ensures that the filtration f is computed on this dataset
	X,Y = graphs.get(dataset = args.dataset, filtration=filtrations[0]) # Fills X and Y
# elif args.datasets == "ModelNet10":
# 	import os
# 	if not os.path.exists()
# 	train, test = shape3d.load_modelnet('10')
# train_graphs, train_labels = shape3d.torch_geometric_2nx(train)
else:
	raise Exception(f"Dataset {args.dataset} not yet supported.")


if args.test:
	from sklearn.preprocessing import LabelEncoder
	Y = LabelEncoder().fit_transform(Y)
	indices = list(np.argwhere(Y==0).squeeze())[:2] + list(np.argwhere(Y==1).squeeze())[:2]
	X=[X[i] for i in indices]
	Y=[Y[i] for i in indices]


print("Classes :", np.unique(Y))






############################## SIMPLEXTREEMULTI PIPELINES
# RD2ST = p2.RipsDensity2SimplexTree(
# 		bandwidth=args.rips_bandwidth,
# 		num_collapse=100, progress=False, sparse=args.sparse_rips,  delayed=False, # Delays the computation to the next pipeline for parallel (simplextree are not pickle-able)
# 		threshold=args.rips_threshold, rescale_density=0,
# 		n_jobs=args.n_jobs
# 		)
RDs2STs = p2.RipsDensity2SimplexTrees(
		bandwidths=bandwidths,
		num_collapses=100, progress=False, sparse=args.sparse_rips,  delayed=False, # Delays the computation to the next pipeline for parallel (simplextree are not pickle-able)
		threshold=args.rips_threshold,
		n_jobs=args.n_jobs
		)

############################# SIGNED MEASURES MULTI PIPELINES
_ST_parameters = {
	"degrees":[None] if (args.topological_invariant == "euler" or args.degrees is None) else args.degrees,
	"num_collapses":100, "progress":True, "n_jobs":args.n_jobs,
	"filtration_quantile":args.drop_quantile, "resolution":args.grid_shape, 
	"sparse": True, 
	"_möbius_inversion": True,
	"normalize_filtrations":False,
	"fit_fraction":1,
	"invariant": "_" if not "hilbert" in args.pipeline else "euler" if args.degrees is None else "hilbert",
	"expand":True,
	"out_resolution": None,
	"infer_filtration_strategy":args.infer_strategy,
	"enforce_null_mass":"smk" in args.pipeline,
}
STM2SM = p2.SimplexTree2SignedMeasure(
		**_ST_parameters
	)
STMs2SMs = p2.SimplexTrees2SignedMeasures(
		**_ST_parameters
	)
DR2SM = p2.DegreeRips2SignedMeasure(
		degrees=args.degrees,
		min_rips_value=0,
		max_rips_value=args.rips_threshold,
		min_normalized_degree=0,
		max_normalized_degree=0.3, # TODO, make a threshold for that
		grid_granularity=args.resolution,
		n_jobs=args.n_jobs, 
		progress=True,
		_möbius_inversion= True,
	)
SMD1 = p2.SignedMeasure2SlicedWassersteinDistance(num_directions=50, n_jobs=args.n_jobs, progress=True)
SMDs = p2.SignedMeasures2SlicedWassersteinDistances(
	num_directions=50, n_jobs=args.n_jobs, 
	progress=True,
	scales = None if args.num_rescales <= 1 else p2.get_filtration_weights_grid(
		resolution=args.num_rescales,
		num_parameters=num_parameters,
		min=0.5,max=1.5,
		remove_homothetie=True,
	)
)
#### NUMBER of axis ?
num_bandwidth=len(bandwidths) if "rd" in args.pipeline else 1
num_kernel_rescale = 1 if SMDs.scales is None else len(SMDs.scales)
num_axes = num_bandwidth*num_kernel_rescale
print(f"Number of axis : bandwidths ({num_bandwidth}) x num_scales ({num_kernel_rescale}) = {num_axes}", flush=1)










############################ VERSION WITH DISCRETE CONVOLUTION : Faster but smaller precision
# SMF = p2.SignedMeasureFormatter(unsparse=True)
# SMF_parameters = {
# 	# "SMF__filtrations_weights": 	[None] if "hilbert" in args.pipeline else p2.get_filtration_weights_grid(num_parameters=num_parameters, weights=[1,.1,10]),
# 	"SMF__axis": 					list(range(num_bandwidth)) if args.pipeline.startswith("rd_") else [None],
# 	"SMF__resolution":				[20, 50, 100] if num_parameters == 2 else [20],
# }
# SMM2CV = p2.SignedMeasure2Convolution(flatten=True, n_jobs=-1)
# print("Num parameters", num_parameters)
# SMM2CV_parameters = {
# 	"SMM2CV__bandwidth":		p2.get_filtration_weights_grid(num_parameters=num_parameters, weights=[1.]+list(np.linspace(.1,10.,args.num_rescales-1)), remove_homothetie=False),
# 	# "SMM2CV__resolution": 		[args.out_resolution],
# 	# "SMM2CV__infer_grid_strategy":	["exact"], # should be fine in every cases
# }

########## VERSION WITH SPARSE CONVOLUTION : Slower but better precision
SMF = p2.SignedMeasureFormatter(unsparse=False, normalize=True)
SMF_parameters = {
	"SMF__filtrations_weights": 	[None] if "hilbert" in args.pipeline else p2.get_filtration_weights_grid(num_parameters=num_parameters, weights=[1.]+list(np.linspace(.1,1.,args.num_rescales-1))),
	"SMF__axis": 					list(range(num_bandwidth)) if args.pipeline.startswith("rd_") else [None],
	# "SMF__resolution":				[20, 50, 100] if num_parameters == 2 else [5],
}
SMM2CV = p2.SignedMeasure2Convolution(flatten=True, n_jobs=-1)
print("Num parameters", num_parameters)
SMM2CV_parameters = {
	"SMM2CV__bandwidth":		[0.01,0.1,1.], ## Normalized filtrations
	"SMM2CV__resolution": 		[args.out_resolution],
	"SMM2CV__infer_grid_strategy":	["exact"], # should be fine in every cases
}



############################## DISTANCE MATRIX MAGIC
D2DL = mmk.DistanceMatrices2DistancesList()
DL2D = mmk.DistancesLists2DistanceMatrices()

DM2K = mmk.DistanceMatrix2Kernel()

DM2K_parameters= {
	"DM2K__sigma":[0.001, 0.01,1, 10, 100, 1000],
	"DM2K__axis":list(range(num_axes)), # if args.pipeline == "rd_smk" else [None], # Only for rips+density
	"DM2K__weights":p2.get_filtration_weights_grid(num_parameters=len(_ST_parameters["degrees"]), weights=[1,.1,10]),
}



SVMP = SVC(kernel = "precomputed")
SVMP_parameters = {
	"SVMP__kernel" : 	["precomputed"],
	"SVMP__C" : 		[0.001, 0.01,1, 10, 100, 1000],
}


########################################### SimplexTree and Diagram Transformers

if args.dataset.startswith("3dshapes/"):
	true_geodesic = args.geodesic_backend == "torch_geometric"
	ToSimplexTree = p1.TorchData2DijkstraSimplexTree(true_geodesic=true_geodesic, progress=True) # dtype=None delays the computation for multithread with simplextrees
	ToSimplexTreeMulti = None # TODO ?
	ToSignedMeasure = None
	SMD=None
elif args.dataset in ["orbit", "immuno"] or args.dataset.startswith("UCR/"): # point clouds
	ToSimplexTree = p1.PointCloud2SimplexTree(threshold=args.diagram_threshold)
	ToSimplexTreeMulti = RDs2STs
	ToSignedMeasure = STMs2SMs
	SMD = SMDs
	 # TODO pop filtrations, ... from args
elif args.dataset.startswith("graphs/") or args.dataset.startswith("ModelNet"):
	ToSimplexTree = p1.Graph2SimplexTree(f=args.filtration)
	ToSimplexTreeMulti = mdg.Graph2SimplexTree(filtrations=args.filtrations)
	# STM2SM.infer_filtration_strategy = "exact"
	STM2SM.num_collapses = 0
	STM2SM.expand = False
	STM2SM.sparse = True
	ToSignedMeasure = STM2SM
	SMD=SMDs
else:
	raise Exception(f"Dataset {args.dataset} not yet supported.")

print("Transformers : ", ToSimplexTree, ToSimplexTreeMulti, ToSignedMeasure, SMD)

print("Initializing diagrams pipeline", flush=True)
# The other pipelines are taking diagrams as an input, so we can factorize the pipeline from here. This allows for multithread computation of the dgms
compute_diagram_pipe = Pipeline([
	("st", ToSimplexTree),
	("dgm", p1.SimplexTree2Dgm(n_jobs=args.n_jobs, threshold=args.diagram_threshold, extended=extended, degrees=degrees, progress=True))
])













# ## Final args
print("Arguments", args)
###########################################PIPELINES PARAMETERS
print("Initializing pipeline", flush=True)
if args.pipeline == "dummy": # Dummy
	pipeline = DummyClassifier()
	parameters = {}

elif args.pipeline == "filvec":
	print("Filvec pipeline")
	svm = SVC(kernel="rbf")
	parameters = {
		"hist__quantile":[0.],
		"hist__bins":[100,200,300],
		"svm__kernel" : ["rbf"],
		"svm__gamma" : [0.01, 0.1, 1, 10, 100],
		"svm__C" : [0.001,0.01,1, 10, 100, 1000],
	}
	pipeline = Pipeline([
		("st",ToSimplexTree), 
		("hist", p1.SimplexTree2Histogram()),
		("svm",svm)
	])
elif args.pipeline == "rd_smi" or args.pipeline ==  "multi_smi":
	### PREPROCESSING : transform to signed measure
	SignedMeasureTransformer = Pipeline([('st', ToSimplexTreeMulti), ("sm",ToSignedMeasure)])
	X = SignedMeasureTransformer.fit_transform(X=X)
	pipeline = Pipeline([
		("SMF", SMF),
		("SMM2CV", SMM2CV),
		("final_classifier", final_classifier),
	],
	# memory=memory
	)
	# SMF_parameters["SMF__filtrations_weights"] = [[1,1]]
	parameters = {}
	parameters.update(SMF_parameters)
	parameters.update(SMM2CV_parameters)
	parameters.update(final_classifier_parameters)
elif args.pipeline == "rd_hilbert" or args.pipeline ==  "multi_hilbert":
	ToSignedMeasure._möbius_inversion = False
	ToSignedMeasure.sparse = False
	SignedMeasureTransformer = Pipeline([('st', ToSimplexTreeMulti), ("sm",ToSignedMeasure)])
	X = SignedMeasureTransformer.fit_transform(X=X)
	SMF.flatten = True
	SMF.unsparse = True
	
	pipeline = Pipeline([
		("SMF", SMF),
		("final_classifier", final_classifier),
	])
	parameters = {}
	parameters.update(SMF_parameters)
	parameters.update(final_classifier_parameters)
elif args.pipeline == "rd_smk" or args.pipeline == "multi_smk":
	ToSignedMeasure.sparse = True
	SMD = SMDs
	SignedMeasureDistancesTransformer = Pipeline([('st', ToSimplexTreeMulti), ("sm",ToSignedMeasure), ("smd", SMD), ("smdl",D2DL)])
	X = SignedMeasureDistancesTransformer.fit_transform(X=X)
	print(f"Num axes of computed measure : {len(X[0])}")
	pipeline = Pipeline([
		("DL2D",DL2D),
		("DM2K",DM2K),
		("SVMP",SVMP),
	])
	parameters = {}
	parameters.update(DM2K_parameters)
	parameters.update(SVMP_parameters)
elif args.pipeline == "dr_smi":
	### Preprocessing: compute the signed measure
	X = DR2SM.fit_transform(X)

	### CLASSIFICATION PIPELINE
	pipeline = Pipeline([
		("SMM2CV", SMM2CV),
		("final_classifier", final_classifier)
	])
	parameters = {}
	parameters.update(SMM2CV_parameters)
	parameters.update(final_classifier_parameters)
elif args.pipeline == "dr_hilbert":
	DR2SM._möbius_inversion = False
	SignedMeasureTransformer = DR2SM
	X = SignedMeasureTransformer.fit_transform(X=X)
	X = [np.reshape(x,-1) for x in X]
	pipeline = Pipeline([
		("final_classifier", final_classifier)
	])
	parameters = {}
	parameters.update(final_classifier_parameters)

elif args.pipeline == "dr_smk":
	### Preprocessing: compute the signed measure distances
	SMD =SMD1
	SignedMeasureTransformer = Pipeline([("DR2SM",DR2SM), ("smd", SMD),  ("D2DL", D2DL)])
	X = SignedMeasureTransformer.fit_transform(X=X)
	## Classification pipeline
	pipeline = Pipeline([
		("dms",DL2D),
		("DM2K", DM2K),
		("SVMP",SVMP)
	])
	parameters = {}
	parameters.update(DM2K_parameters)
	parameters.update(SVMP_parameters)

elif args.pipeline == "sw": # Diagram -> SW -> SVM
	svm = SVC(kernel = "precomputed")
	print("Computing Sliced Wassertstein Distances", flush=True)
	diagrams = compute_diagram_pipe.fit_transform(X)
	swds = p1.Dgms2SlicedWassersteinDistanceMatrices(num_directions=10, n_jobs=args.n_jobs).fit_transform(diagrams)
	print("Formatting Distance Matrix", flush=True)
	X = p1.DistanceMatrices2DistancesList().fit_transform(swds)
	pipeline = Pipeline([
		("dms",DL2D),
		("DM2K", DM2K),
		("SVMP",SVMP)
	])
	parameters = {}
	DM2K_parameters.pop("DM2K__axis")
	DM2K_parameters.pop("DM2K__weights")
	parameters.update(DM2K_parameters)
	parameters.update(SVMP_parameters)
# elif args.pipeline == "sw_p": # Graph -> SimplexTree -> Diagram -> Shuffled Diagram -> SW -> SVM
# 	svm = SVC(kernel = "precomputed")
# 	params={
# 		"sw__bandwidth":[0.01, 0.1, 1, 10, 100],
# 		"sw__num_directions":[10],
# 		"svm__kernel" : ["precomputed"],
# 		"svm__C" : [0.01,1, 10, 100, 1000],
# 	}
# 	pipe = Pipeline([
# 		("shuffle", DiagramShuffle()),
# 		("sw", Dgms2SWK()),
# 		("svm",svm)
# 	])
elif args.pipeline == "pervec": # Graph -> SimplexTree -> Diagram -> Diagram values Histogram -> Gaussian SVM
	X = compute_diagram_pipe.fit_transform(X)
	svm = SVC(kernel="rbf")
	parameters = {
		"hist__quantile":[0.],
		"hist__bins":[100,200,300],
		"svm__kernel" : ["rbf"],
		"svm__gamma" : [0.01, 0.1, 1, 10, 100],
		"svm__C" : [0.001,0.01,1, 10, 100, 1000],
	}
	pipeline = Pipeline([
		("hist", p1.Dgm2Histogram()),
		("svm",svm)
	])
elif args.pipeline == "smk": # Graph -> SimplexTree -> Diagram -> Signed measure -> Signed Measure Kernel
	diagrams = compute_diagram_pipe.fit_transform(X)
	# print(diagrams)
	smds = p1.Dgms2SignedMeasureDistance(n_jobs=args.n_jobs, progress=True).fit_transform(diagrams)
	# print(smds)
	X = p1.DistanceMatrices2DistancesList().fit_transform(smds)
	pipeline = Pipeline([
		("dms",DL2D),
		("DM2K", DM2K),
		("SVMP",SVMP)
	])
	parameters = {}
	DM2K_parameters.pop("DM2K__axis")
	DM2K_parameters.pop("DM2K__weights")
	parameters.update(DM2K_parameters)
	parameters.update(SVMP_parameters)
elif args.pipeline == "smh": # Graph -> SimplexTree -> Diagram -> Signed measure -> Signed Measure histogram
	X = compute_diagram_pipe.fit_transform(X)
	hist = p1.Dgms2SignedMeasureHistogram()
	svm = SVC(kernel="rbf")
	pipeline = Pipeline([
		("hist", hist),
		("svm",svm),
	])
	parameters = {
		"hist__quantile" : [0,0.01,0.1],
		"hist__bins":[50,100,200,300],
		"svm__kernel" : ["rbf"],
		"svm__gamma" : [0.01, 0.1, 1, 10, 100],
		"svm__C" : [0.001,0.01,1, 10, 100, 1000],
	}
elif args.pipeline == "smi": # Graph -> SimplexTree -> Diagram -> Signed measure -> Signed Measure image
	X = compute_diagram_pipe.fit_transform(X)
	img = p1.Dgms2SignedMeasureImage()
	svm = SVC(kernel="rbf")
	pipeline = Pipeline([
		("img", img),
		("svm",svm),
	])
	parameters = {
		"img__quantile" : [0,0.01,0.1],
		"img__bandwidth" : [0.01, 0.1, 1., 10., 100.],
		"img__resolution":[50,100,200,300],
		"svm__kernel" : ["rbf"],
		"svm__gamma" : [0.01, 0.1, 1, 10, 100],
		"svm__C" : [0.001,0.01,1, 10, 100, 1000],
	}
elif args.pipeline == "pl": #Landscapes
	X = compute_diagram_pipe.fit_transform(X)
	pipeline = Pipeline([("pl", p1.Dgms2Landscapes()), ("svm", SVC(kernel="rbf"))])
	parameters = {
		"svm__kernel" : ["rbf"],
		"svm__gamma" : [0.01, 0.1, 1, 10, 100],
		"svm__C" : [0.001,0.01,1, 10, 100, 1000],
		"pl__num": [3,4,5,6,7,8], #num landscapes
		"pl__resolution": [50,100,200,300],
	}
elif args.pipeline == "pl_p": # Shuffled Landscapes
	X = compute_diagram_pipe.fit_transform(X)
	pipeline = Pipeline([("shuffle", p1.DiagramShuffle()),("pl", p1.Dgms2Landscapes()), ("svm", SVC(kernel="rbf"))])
	parameters = {
		"svm__kernel" : ["rbf"],
		"svm__gamma" : [0.01, 0.1, 1, 10, 100],
		"svm__C" : [0.001,0.01,1, 10, 100, 1000],
		"pl__num": [3,4,5,6,7,8], #num landscapes
		"pl__resolution": [50,100,200,300],
	}
elif args.pipeline == "pi": # Images
	X = compute_diagram_pipe.fit_transform(X)
	pipeline = Pipeline([("pi", p1.Dgms2Image()), ("svm", SVC(kernel="rbf"))])
	parameters = {
		"svm__kernel" : ["rbf"],
		"svm__gamma" : [0.01, 0.1, 1, 10, 100],
		"svm__C" : [0.001,0.01,1, 10, 100, 1000],
		"pi__bandwidth": [0.01,0.1,1,10,100], 
		"pi__resolution": [[20,20], [30,30]],
	}
elif args.pipeline == "pi_p": # Shuffled Immages
	X = compute_diagram_pipe.fit_transform(X)
	pipeline = Pipeline([("shuffle", p1.DiagramShuffle()),("pi", p1.Dgms2Image()), ("svm", SVC(kernel="rbf"))])
	parameters = {
		"svm__kernel" : ["rbf"],
		"svm__gamma" : [0.01, 0.1, 1, 10, 100],
		"svm__C" : [0.001,0.01,1, 10, 100, 1000],
		"pi__bandwidth": [0.01,0.1,1,10,100], 
		"pi__resolution": [[20,20], [30,30]],
	}	
else:
	raise Exception(f"Pipeline {args.pipeline} not supported.")

print("Initializing classification pipeline", flush=True)
from sklearn.model_selection import GridSearchCV
classifier = GridSearchCV(
	estimator=pipeline, param_grid=parameters,
	n_jobs = args.n_jobs, cv=args.train_k, verbose=1)

######################################SCORE
print("Computing classification, with pipeline", flush=True)
print(pipeline, flush=True)
print("Final parameters : ", parameters)


# try: 
# 	filtration_grid = ToSignedMeasure.filtration_grid
# 	print("Signed Measure Filtration grid : ", filtration_grid)
# except:
# 	None

from multipers.ml.accuracies import accuracy_to_csv
from joblib import parallel_backend
with parallel_backend('threading', n_jobs=args.n_jobs):
	accuracy_to_csv(
		X=X, Y=Y, dataset = dataset, cl=classifier, k=args.test_k,
		shuffle = shuffle,
		**results_kwargs
	)
# os.system(f"rm -rf {memory}") # removes cache
