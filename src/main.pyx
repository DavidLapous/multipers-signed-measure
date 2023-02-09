# distutils: language = c++
# distutils: include_dirs = mma_cpp, gudhi, rank_invariant

###########################################################################
## PYTHON LIBRARIES
import gudhi as gd
import numpy as np
from typing import List, Union
from os.path import exists
from os import remove 
from tqdm import tqdm 
from cycler import cycler
from joblib import Parallel, delayed
import pickle as pk

###########################################################################
## CPP CLASSES
from cython.operator import dereference, preincrement
from libc.stdint cimport intptr_t
from libc.stdint cimport uintptr_t

###########################################################################
## CYTHON TYPES
from cython cimport numeric
from libcpp.vector cimport vector
from libcpp.utility cimport pair
#from libcpp.list cimport list as clist
from libcpp cimport bool
from libcpp cimport int
from libcpp.string cimport string


#########################################################################

ctypedef double value_type # type of simplextrees filtrations vector

ctypedef vector[pair[int,pair[int,int]]] barcode
ctypedef vector[pair[int,pair[value_type,value_type]]] barcoded
ctypedef vector[unsigned int] boundary_type
ctypedef vector[boundary_type] boundary_matrix
ctypedef pair[pair[value_type,value_type],pair[value_type,value_type]] interval_2
ctypedef vector[value_type] filtration_type
ctypedef vector[filtration_type] multifiltration
ctypedef vector[int] simplex_type
ctypedef int dimension_type

#########################################################################
## Small hack 
from gudhi import SimplexTree as GudhiSimplexTree
from multipers import SimplexTree as MSimplexTree


###########################################################################
#PYX MODULES

### Multiparameter simplextrees 
include "simplex_tree_multi.pyx"


### Rank invariant over simplex trees
include "rank_invariant.pyx"









