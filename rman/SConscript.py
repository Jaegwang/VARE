##------------##
## SConscript ##
##-------------------------------------------------------##
## author: Jaegwang Lim @ Dexter Studios                 ##
## last update: 2019.02.21                               ##
##-------------------------------------------------------##

import SCons.Tool
import SCons.Defaults
import os, sys
from distutils.dir_util import copy_tree

sys.path.insert( 0, os.path.abspath("..") )
sys.path.insert( 0, os.path.abspath("../..") )
sys.path.insert( 0, os.path.abspath("../../..") )
from SCGlobal import *

RMAN_VER = os.environ['RMAN_VER']
RMAN_INSTALL_PATH = '/opt/RenderManProServer-'+RMAN_VER

#################
# C++ Compile   #
#################
cppenv = Environment()
cppenv['ENV']['TERM'] = os.environ['TERM']
cppenv['STATIC_AND_SHARED_OBJECTS_ARE_THE_SAME'] = 1

cppenv.Append( CCFLAGS = '-Wall -W -O3 -m64 -fPIC -fopenmp -std=c++11' )
cppenv.Append( CCFLAGS = SWITCHES )

cppenv.Append( CPPPATH = INCLUDE_PATH )
cppenv.Append( CPPPATH = [
    '../../../base/header',
    RMAN_INSTALL_PATH+'/include',
])

cppenv.Append( LIBPATH = LIBRARY_PATH )
cppenv.Append( LIBPATH = [
    '../../lib',
    RMAN_INSTALL_PATH+'/lib'
])

cppenv.Append( LIBS = LIBRARY_LIST )
cppenv.Append( LIBS = [
    'BoraBase',
    'prman',
    'prman-'+RMAN_VER
])

cppenv.Append( CPPDEFINES = DEFINE_LIST )

#cppenv.Object( Glob('BoraRmanVDBFromBgeo/*.cpp') )
#cppenv.SharedLibrary( 'BoraRmanVDBFromBgeo', Glob('BoraRmanVDBFromBgeo/*.o') )

cppenv.Object( Glob('PxrBoraOcean/*.cpp') )
cppenv.SharedLibrary( 'PxrBoraOcean', Glob('PxrBoraOcean/*.o') )

