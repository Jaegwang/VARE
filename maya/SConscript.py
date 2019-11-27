##------------##
## SConscript ##
##-------------------------------------------------------##
## author: Jaegwang Lim @ Dexter Studios                 ##
## last update: 2018.12.05                               ##
##-------------------------------------------------------##

import SCons.Tool
import SCons.Defaults
import os, sys
from distutils.dir_util import copy_tree

sys.path.insert( 0, os.path.abspath("..") )
sys.path.insert( 0, os.path.abspath("../..") )
sys.path.insert( 0, os.path.abspath("../../..") )
from SCGlobal import *

MAYA_VER = os.environ['MAYA_VER']
MAYA_INSTALL_PATH = '/usr/autodesk/maya'+MAYA_VER

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
    MAYA_INSTALL_PATH+'/include',    
])

cppenv.Append( LIBPATH = LIBRARY_PATH )
cppenv.Append( LIBPATH = [
    '../../lib',
    MAYA_INSTALL_PATH+'/lib'
])

cppenv.Append( LIBS = LIBRARY_LIST )
cppenv.Append( LIBS = [
    'BoraBase',
    'OpenMaya',
    'OpenMayaAnim',
    'Foundation',
    'OpenMayaUI',
    'OpenMayaFX'
])

cppenv.Append( CPPDEFINES = DEFINE_LIST )

cppenv.Object( Glob('source/*.cpp') )
cppenv.SharedLibrary( 'plug-ins/'+'BoraForMaya', Glob('source/*.o') )

#################
# File Transfer #
#################
copy_tree( '../../../maya/script', 'scripts' )

