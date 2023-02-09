

__author__ = "David Loiseaux, TODO"
__copyright__ = "Copyright (C) 2023  Inria"
__license__ = ""

# from distutils.core import setup
# from distutils.extension import Extension
from os.path import exists
from multiprocessing import cpu_count
from setuptools import Extension, setup
from Cython.Build import cythonize

from Cython.Compiler import Options

Options.docstrings = True
Options.embed_pos_in_docstring = True
Options.fast_fail = True

## Regenerate cpp files using Cython
USE_CYTHON=True
# USE_CYTHON = not exists("main.cpp")

cython_compiler_directives = {
    "language_level": 3,
    "embedsignature": True,
    "binding": True,
    "infer_types": True,
    # "show_all_warnings": True,
    # "nthreads":max(1, cpu_count()//2),
}

cythonize_flags = {
    # "depfile":True,
    "nthreads": (int)(max(1, cpu_count()//2)),
    # "show_all_warnings":True,
}

source_files = [
	"main.pyx" if USE_CYTHON else "main.cpp"
]

extensions = [Extension('multipers',
                        sources=source_files,
                        language='c++',
                        extra_compile_args=[
                            "-O3",
                            "-march=native",
                            # "-g0",
                            "-std=c++20",
                            '-fopenmp',
                            "-Wall"
                        ],
                        extra_link_args=['-fopenmp'],
                        )]
setup(
    name='multipers',
    author="David Loiseaux",
    author_email="david.loiseaux@inria.fr",
    description="Multiparameter persistence toolkit",
	ext_modules=cythonize(
		extensions, compiler_directives=cython_compiler_directives, **cythonize_flags),
	include_dirs = ['.', "gudhi", "rank_invariant"],
)
