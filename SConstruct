##------------##
## SConstruct ##
##-------------------------------------------------------##
## author: Jaegwang Lim @ Dexter Studios                 ##
## last update: 2019.02.21                               ##
##-------------------------------------------------------##

import os, sys

all = ARGUMENTS.get( 'all', 0 )

if all == 0 :

    ## Bora Base --------------------------------------------##

    if not os.path.exists( 'build/lib') : os.makedirs( 'build/lib' )
    SConscript( 'base/SConscript.py', variant_dir='build/lib', duplicate=0 )


    ## Bora For Maya ----------------------------------------##

    MAYA_VER = os.environ.get( 'MAYA_VER', '2018' )
    os.environ['MAYA_VER'] = MAYA_VER

    if not os.path.exists( 'build/maya/'+MAYA_VER) : os.makedirs( 'build/maya/'+MAYA_VER )
    SConscript( 'maya/SConscript.py', variant_dir='build/maya/'+MAYA_VER, duplicate=0 )


    ## Bora For Houdini -------------------------------------##

    HOU_VER = os.environ.get( 'HOU_VER', '17.0.352' )
    os.environ['HOU_VER'] = HOU_VER

    if not os.path.exists( 'build/houdini/'+HOU_VER) : os.makedirs( 'build/houdini/'+HOU_VER )
    SConscript( 'houdini/SConscript.py', variant_dir='build/houdini/'+HOU_VER, duplicate=0 )

else :

    print '-------------------------------------------------'
    print ' Dexter Studios - Digital '
    print ' Bora : Compiling In-House Tools... '
    print '-------------------------------------------------'

    ## Bora Base --------------------------------------------##

    if not os.path.exists( 'build/lib') : os.makedirs( 'build/lib' )
    SConscript( 'base/SConscript.py', variant_dir='build/lib', duplicate=0 )


    ## Bora For Maya ----------------------------------------##

    os.environ['MAYA_VER'] = '2018'
    if not os.path.exists( 'build/maya/2018') : os.makedirs( 'build/maya/2018' )
    SConscript( 'maya/SConscript.py', variant_dir='build/maya/2018', duplicate=0 )

    os.environ['MAYA_VER'] = '2017'
    if not os.path.exists( 'build/maya/2017') : os.makedirs( 'build/maya/2017' )
    SConscript( 'maya/SConscript.py', variant_dir='build/maya/2017', duplicate=0 )

    os.environ['MAYA_VER'] = '2016.5'
    if not os.path.exists( 'build/maya/2016.5') : os.makedirs( 'build/maya/2016.5' )
    SConscript( 'maya/SConscript.py', variant_dir='build/maya/2016.5', duplicate=0 )


    ## Bora For Houdini -------------------------------------##

    os.environ['HOU_VER'] = '17.0.352'
    if not os.path.exists( 'build/houdini/17.0.352') : os.makedirs( 'build/houdini/17.0.352' )
    SConscript( 'houdini/SConscript.py', variant_dir='build/houdini/17.0.352', duplicate=0 )

    os.environ['HOU_VER'] = '16.5.571'
    if not os.path.exists( 'build/houdini/16.5.571') : os.makedirs( 'build/houdini/16.5.571' )
    SConscript( 'houdini/SConscript.py', variant_dir='build/houdini/16.5.571', duplicate=0 )


    ## Bora For RenderMan -----------------------------------##

    #os.environ['RMAN_VER'] = '22.3'
    #if not os.path.exists( 'build/rman/22.3') : os.makedirs( 'build/rman/22.3' )
    #SConscript( 'rman/SConscript.py', variant_dir='build/rman/22.3', duplicate=0 )

