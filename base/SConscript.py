##------------##
## SConscript ##
##-------------------------------------------------------##
## author: Jaegwang Lim @ Dexter Studios                 ##
## last update: 2019.04.01                               ##
##-------------------------------------------------------##

import SCons.Tool
import SCons.Defaults
import os, sys
import os.path, time
from distutils.dir_util import copy_tree

sys.path.insert( 0, os.path.abspath("..") )
sys.path.insert( 0, os.path.abspath("../..") )
from SCGlobal import *

cudaenv = Environment()

cuda_static, cuda_shared = SCons.Tool.createObjBuilders( cudaenv )
cuda_static.add_action( '.cu', SCons.Defaults.CXXAction )
cuda_shared.add_action( '.cu', SCons.Defaults.CXXAction )
#cuda_static.add_action( '.cu', SCons.Defaults.StaticObjectEmitter )
#cuda_shared.add_action( '.cu', SCons.Defaults.SharedObjectEmitter )

cudaenv['ENV']['TERM'] = os.environ['TERM']
cudaenv['STATIC_AND_SHARED_OBJECTS_ARE_THE_SAME'] = 1

cudaenv['CC'] = '/usr/local/cuda-8.0/bin/nvcc'
cudaenv['CXX'] = '/usr/local/cuda-8.0/bin/nvcc'

cudaenv.Append( CCFLAGS = '-m64 -arch=compute_62 -Xcompiler -fPIC -O3 -std=c++11 --disable-warnings --expt-extended-lambda --default-stream per-thread' )

cudaenv.Append( CPPPATH = INCLUDE_PATH )
cudaenv.Append( CPPPATH = [ 'unittest' ] )

cudaenv.Append( LIBPATH = LIBRARY_PATH )

cudaenv.Append( LIBS = LIBRARY_LIST )
cudaenv.Append( LIBS = [
	'cudart',
	'cublas',
	'GL',
    'GLU',
    'GLEW',
    'glfw',
	'tbb',
    'openvdb',
    'BulletDynamics',
	'BulletCollision',
    'Bullet3Dynamics',
	'Bullet3Collision',
	'Bullet3Common',
	'LinearMath'
])

cudaenv.Append( CPPDEFINES = DEFINE_LIST )

cudaenv.Object( Glob('source/*.cu') )
cudaenv.Object( Glob('source/*.cpp') )

cudaenv.SharedLibrary( 'BoraBase', Glob('source/*.o') )

# Unit-test compile
#cudaenv['RPATH'] = cudaenv['LIBPATH']
cudaenv.Object( Glob('unittest/*.cpp') )
cudaenv.Object( Glob('unittest/*.cu') )

objs = Glob('source/*.o')
objs.append( Glob('unittest/*.o') )

cudaenv.Program( 'test', objs )

