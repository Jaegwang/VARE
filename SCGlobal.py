##-------------##
## SCGlobal.py ##
##-------------------------------------------------------##
## author: Jaegwang Lim @ Dexter Studios                 ##
##         Wanho Choi @ Dexter Studios                   ##
## last update: 2018.10.31                               ##
##-------------------------------------------------------##

import os, sys

INCLUDE_PATH = [
    '.',
    './header',
    '/usr/local/include',
    '/usr/local/include/hdf5',
    '/usr/local/include/OpenEXR',
    '/usr/local/include/bullet',
    '/usr/local/cuda-8.0/include'    
]

LIBRARY_PATH = [
    '.',
    '/usr/local/lib',
    '/usr/local/cuda-8.0/lib64'
]

LIBRARY_LIST = [
    'Half',
    'boost_filesystem',
    'boost_system',
    'boost_iostreams',
    'IlmThread',
    'IlmImf',
    'jpeg',
    'fftw3f',
    'fftw3f_threads',
    'hdf5'
]

DEFINE_LIST = [
    'GLM_FORCE_CUDA',
    'GLM_ENABLE_EXPERIMENTAL',
    '_GNU_SOURCE',
    'LINUX',
    'AMD64',    
    'BT_USE_DOUBLE_PRECISION',
    'PYBULLET_USE_NUMPY',
    'USE_GRAPHICAL_BENCHMARK',
    'CUDA_NO_HALF'
]

SWITCHES = [
    '-Wno-deprecated',
    '-Wno-parentheses',
    '-Wno-sign-compare',
    '-Wno-strict-aliasing',
    '-Wno-reorder',
    '-Wno-uninitialized',
    '-Wno-unused',
    '-Wno-unused-parameter',
    '-Wno-unused-local-typedefs',
    '-Wno-unused-variable',
    '-Wno-write-strings',
    '-Wno-overflow',
    '-Wno-ignored-qualifiers'
]

