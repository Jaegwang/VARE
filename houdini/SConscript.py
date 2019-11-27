##------------##
## SConscript ##
##-------------------------------------------------------##
## author: Jaegwang Lim @ Dexter Studios                 ##
## last update: 2019.04.10                               ##
##-------------------------------------------------------##

import SCons.Tool
import SCons.Defaults
import os, sys
import os.path, time
import shutil
from distutils.dir_util import copy_tree

sys.path.insert( 0, os.path.abspath("..") )
sys.path.insert( 0, os.path.abspath("../..") )
sys.path.insert( 0, os.path.abspath("../../..") )
from SCGlobal import *

HOU_VER = os.environ['HOU_VER']
HOU_INSTALL_PATH = '/opt/hfs'+HOU_VER

#################
# C++ Compile   #
#################
cppenv = Environment()
cppenv['ENV']['TERM'] = os.environ['TERM']
cppenv['STATIC_AND_SHARED_OBJECTS_ARE_THE_SAME'] = 1

cppenv.Append( CCFLAGS = '-Wall -W -O2 -m64 -fPIC -shared -std=c++11' )
cppenv.Append( CCFLAGS = SWITCHES )

cppenv.Append( CPPPATH = INCLUDE_PATH )
cppenv.Append( CPPPATH = [
    '../../../base/header',
    HOU_INSTALL_PATH+'/toolkit/include',
])

cppenv.Append( LIBPATH = LIBRARY_PATH )
cppenv.Append( LIBPATH = [
    '../../lib',
    HOU_INSTALL_PATH+'/dsolib',
])

cppenv.Append( LIBS = LIBRARY_LIST )
cppenv.Append( LIBS = [
    'BoraBase',
    'openvdb_sesi',
    'HoudiniAPPS1',
    'HoudiniAPPS2',
    'HoudiniAPPS3',
    'HoudiniDEVICE',
    'HoudiniGEO',
    'HoudiniOP1',
    'HoudiniOP2',
    'HoudiniOP3',
    'HoudiniOPZ',
    'HoudiniPDG', 
    'HoudiniOPZ',
    'HoudiniPRM',
    'HoudiniRAY',
    'HoudiniSIM',
    'HoudiniUI',
    'HoudiniUT',
    'HoudiniHAPIL',
    'HoudiniHAPI',
    'HoudiniHAPI',
    'HoudiniHARC',
    'HoudiniHARD'
])

cppenv.Append( CPPDEFINES = DEFINE_LIST )
cppenv.Append( CPPDEFINES = [
    'UT_DSO_TAGINFO=\\"k3498yugjb18746y7dfbn392\\"',
    'VERSION=\\"'+HOU_VER+'\\"',
    'SIZEOF_VOID_P=8',
    'FBX_ENABLED=1',
    'OPENCL_ENABLED=1',
    'OPENVDB_ENABLED=1',
    'SESI_LITTLE_ENDIAN',
    'ENABLE_THREADS',
    'USE_PTHREADS',
    '_REENTRANT',
    '_FILE_OFFSET_BITS=64',
    'GCC4',
    'GCC3',
    'MAKING_DSO'
])

cppenv.Object( Glob('source/*.cpp') )
cppenv.SharedLibrary( 'plugins/'+'BoraForHoudini', Glob('source/*.o') )

#print("BoraForHoudini compiled on : %s" % time.ctime(os.path.getmtime("plugins/libBoraForHoudini.so")))

#################
# File Transfer #
#################
os.system( 'rm -rf otls' )
shutil.copytree( '../../../houdini/otls', 'otls' )

