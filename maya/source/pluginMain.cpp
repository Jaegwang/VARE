//----------------//
// PluginMain.cpp //
//-------------------------------------------------------//
// author: Jaegwwang Lim @ Dexter Studios                //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2019.01.23                               //
//-------------------------------------------------------//

#include <MayaCommon.h>
#include <MayaRegister.h>
#include <MayaViewOverride.h>
#include <maya/MFnPlugin.h>

#include <Bora_ConvertToLevelSet_FSM.h>
#include <Bora_ConvertToLevelSet_FMM.h>
#include <Bora_MarchingCubesTest.h>

#include <BoraNodeData.h>

#include <BoraNodeTemplate.h>
#include <BoraGrid.h>

#include <BoraOcean.h>
#include <BoraOceanCmd.h>

#include <BoraBreakingWave.h>
#include <BoraOceanMesh.h>
#include <BoraHeightMerge.h>
#include <BoraTest.h>

MStatus initializePlugin( MObject obj )
{
    if( MGlobal::mayaState() != MGlobal::kBatch ) glewInit();

	MStatus s = MS::kSuccess;    
	MFnPlugin pluginFn( obj, "Dexter Studios", "1.0", "Any" );

    //////////////
	// Commands //
    RegisterCommand( pluginFn, BoraOceanCmd );

    //////////
    // Data //
	RegisterData( pluginFn, BoraNodeData );

    /////////////
	// DG Node //
    RegisterNode( pluginFn, BoraBreakingWave );
    RegisterNode( pluginFn, BoraOceanMesh );

    ///////////////////
	// Locator Nodes //
    RegisterLocatorWithDrawOverride( pluginFn, BoraNodeTemplate, BoraNodeTemplateDrawOverride );
    RegisterLocatorWithDrawOverride( pluginFn, BoraGrid, BoraGridDrawOverride );
    RegisterLocatorWithDrawOverride( pluginFn, BoraOcean, BoraOceanDrawOverride );
    RegisterLocatorWithDrawOverride( pluginFn, BoraTest, BoraTestDrawOverride );
    RegisterLocatorWithDrawOverride( pluginFn, BoraHeightMerge, BoraHeightMergeDrawOverride );

    ////////////
	// Shapes //
	RegisterShapeWithDrawOverride( pluginFn, Bora_ConvertToLevelSet_FSM, Bora_ConvertToLevelSet_FSMDrawOverride );
	RegisterShapeWithDrawOverride( pluginFn, Bora_ConvertToLevelSet_FMM, Bora_ConvertToLevelSet_FMMDrawOverride );
	RegisterShapeWithDrawOverride( pluginFn, Bora_MarchingCubesTest, Bora_MarchingCubesTestDrawOverride );

    /////////////
	// scripts //
	MGlobal::sourceFile( "BoraUtils.mel" );
	MGlobal::sourceFile( "BoraMenu.mel"  );
	MGlobal::sourceFile( "BoraOcean.mel" );

    //////////
    // menu //
	pluginFn.registerUI( "CreateBoraMenu", "DeleteBoraMenu" );

    return MS::kSuccess;
}

MStatus uninitializePlugin( MObject obj )
{
	MStatus s = MS::kSuccess;
	MFnPlugin pluginFn( obj );

    //////////////
	// Commands //
    DeregisterCommand( pluginFn, BoraOceanCmd );

    //////////
    // Data //
    DeregisterData( pluginFn, BoraNodeData );

    /////////////
	// DG Node //
    DeregisterNode( pluginFn, BoraBreakingWave );
    DeregisterNode( pluginFn, BoraOceanMesh );
    DeregisterNode( pluginFn, BoraHeightMerge );
    

    ///////////////////
	// Locator Nodes //
    DeregisterLocatorWithDrawOverride( pluginFn, BoraNodeTemplate );
    DeregisterLocatorWithDrawOverride( pluginFn, BoraGrid );
    DeregisterLocatorWithDrawOverride( pluginFn, BoraOcean );
    DeregisterLocatorWithDrawOverride( pluginFn, BoraTest );
    DeregisterLocatorWithDrawOverride( pluginFn, BoraHeightMerge );

    ////////////
    // Shapes //
	DeregisterShapeWithDrawOverride( pluginFn, Bora_ConvertToLevelSet_FSM );
	DeregisterShapeWithDrawOverride( pluginFn, Bora_ConvertToLevelSet_FMM );
	DeregisterShapeWithDrawOverride( pluginFn, Bora_MarchingCubesTest );

    return MS::kSuccess;
}

