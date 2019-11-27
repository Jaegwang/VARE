##--------##
## run.py ##
##-------------------------------------------------------##
## author: Jaegwang Lim @ Dexter Studios                 ##
## last update: 2018.10.15                               ##
##-------------------------------------------------------##

import os, sys

sys.path.insert( 0, os.path.abspath("..") )
sys.path.insert( 0, os.path.abspath("../..") )
from SCGlobal import *

HOU_VER = os.environ.get( 'HOU_VER', '17.0.352' )
HOU_INSTALL_PATH = '/opt/hfs'+HOU_VER

### BORA HOUDINI PLUGINS
os.environ['HOUDINI_DSO_ERROR']='2'
os.environ['HOUDINI_VERBOSE_ERROR']='1'
os.environ['HOUDINI_TEXT_CONSOLE']='1'

os.environ['HOUDINI_DSO_PATH']  = '&'
os.environ['HOUDINI_DSO_PATH'] += ':'+os.path.abspath('../build/houdini/'+HOU_VER+'/plugins')
os.environ['HOUDINI_DSO_PATH'] += ':'+os.path.abspath('./build/houdini/'+HOU_VER+'/plugins')

os.environ['HOUDINI_VEX_DSO_PATH']  = '&'
os.environ['HOUDINI_VEX_DSO_PATH'] += ':'+os.path.abspath('../build/houdini/'+HOU_VER+'/plugins')
os.environ['HOUDINI_VEX_DSO_PATH'] += ':'+os.path.abspath('./build/houdini/'+HOU_VER+'/plugins')

os.environ['HOUDINI_OTLSCAN_PATH']  = '&'
os.environ['HOUDINI_OTLSCAN_PATH'] += ':./houdini/otls'
os.environ['HOUDINI_OTLSCAN_PATH'] += ':./otls'

os.environ['HOUDINI_TOOLBAR_PATH']  = '&'

os.environ['LD_LIBRARY_PATH'] += ':'+HOU_INSTALL_PATH+'/dsolib'
os.environ['LD_LIBRARY_PATH'] += ':'+os.path.abspath('../build/lib')
os.environ['LD_LIBRARY_PATH'] += ':'+os.path.abspath('./build/lib')
os.environ['LD_LIBRARY_PATH'] += ':/usr/local/lib'
os.environ['LD_LIBRARY_PATH'] += ':/usr/local/cuda/lib64'

### END ###

### DX_PipelineTools-2.2
os.environ['HOUDINI_OTLSCAN_PATH'] += ':/netapp/backstage/pub/apps/houdini2/tools/DX_pipelineTools-2.2/otls'
os.environ['PYTHONPATH'] += ':/netapp/backstage/pub/apps/houdini2/tools/DX_pipelineTools-2.2/scripts'

### DXK_renderfarm
os.environ['HOUDINI_OTLSCAN_PATH'] += ':/netapp/backstage/pub/apps/houdini2/tools/DXK_renderfarm/otls'
os.environ['PYTHONPATH'] += ':/netapp/backstage/pub/apps/houdini2/tools/DXK_renderfarm/scripts'

### DXK_pipelineTools-0.16
os.environ['HOUDINI_OTLSCAN_PATH'] += ':/netapp/backstage/pub/apps/houdini2/tools/DXK_pipelineTools-0.16/otls'
os.environ['PYTHONPATH'] += ':/netapp/backstage/pub/apps/houdini2/tools/DXK_pipelineTools-0.16/scripts'

### DXK_renderTools-0.1
os.environ['HOUDINI_OTLSCAN_PATH'] += '/netapp/backstage/pub/apps/houdini2/tools/DXK_renderTools-0.1/otls'
os.environ['PYTHONPATH'] += ':/netapp/backstage/pub/apps/houdini2/tools/DXK_renderTools-0.1/scripts'

### DXK_old_Otls-0.1
os.environ['HOUDINI_OTLSCAN_PATH'] += ':/netapp/backstage/pub/apps/houdini2/tools/DXK_old_Otls-0.1/otls'
os.environ['HOUDINI_TOOLBAR_PATH'] += ':/netapp/backstage/pub/apps/houdini2/tools/DXK_old_Otls-0.1/toolbar'
os.environ['PYTHONPATH'] += ':/netapp/backstage/pub/apps/houdini2/tools/DXK_old_Otls-0.1/scripts'

### DXK_sceneSetupMng-0.2
os.environ['HOUDINI_OTLSCAN_PATH'] += ':/netapp/backstage/pub/apps/houdini2/tools/DXK_sceneSetupMng-0.2/otls'
os.environ['HOUDINI_TOOLBAR_PATH'] += ':/netapp/backstage/pub/apps/houdini2/tools/DXK_sceneSetupMng-0.2/toolbar'
os.environ['PYTHONPATH'] += ':/netapp/backstage/pub/apps/houdini2/tools/DXK_sceneSetupMng-0.2/scripts'

os.environ['HB'] = HOU_INSTALL_PATH+'/bin'
os.environ['PATH'] += ':'+HOU_INSTALL_PATH+'/bin'

## PYTHON Packages
os.environ['PYTHONPATH'] += ':/netapp/backstage/pub/apps/tractor/linux/Tractor-2.2/lib/python2.7/site-packages'

## Print to Debug
#os.system( 'echo $LD_LIBRARY_PATH' )
#os.system( 'ldd '+os.path.abspath('./build/lib')+'/libBoraBase.so' )

### Execute Houdini-FX
os.system( HOU_INSTALL_PATH+'/bin/houdinifx' )

