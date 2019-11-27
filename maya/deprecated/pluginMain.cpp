//----------------//
// PluginMain.cpp //
//-------------------------------------------------------//
// author: Jaegwwang Lim @ Dexter Studios                //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2017.11.21                               //
//-------------------------------------------------------//

#include <MayaCommon.h>
#include <MayaRegister.h>
#include <MayaViewOverride.h>
#include <maya/MFnPlugin.h>

#include <BoraData.h>
#include <BoraBgeoImporter.h>
#include <BoraVDBFromPoints.h>
#include <BoraVDBFromPointsCmd.h>

#include <Bora_Grid.h>

MStatus initializePlugin( MObject obj )
{
	MStatus stat = MS::kSuccess;    
	MFnPlugin pluginFn( obj, "Dexter Studios", "1.0", "Any" );

	// Node data
	RegisterData( BoraData );
	
	// Locator Nodes
	RegisterLocatorNodeWithDrawOverride( BoraBgeoImporter, MayaViewOverride<BoraBgeoImporter> );
	RegisterLocatorNodeWithDrawOverride( BoraVDBFromPoints, MayaViewOverride<BoraVDBFromPoints> );	
	
	// Commands
	RegisterCommand( BoraVDBFromPointsCmd );

	RegisterShapeWithDrawOverride( Bora_Grid, Bora_GridDrawOverride );


	/*
	// .mel scripts
	MGlobal::sourceFile( "ZelosMenu.mel"        );
	MGlobal::sourceFile( "ZelosUtils.mel"       );
	MGlobal::sourceFile( "ZarVisUtils.mel"      );
	MGlobal::sourceFile( "ZVOceanWave.mel"      );
	MGlobal::sourceFile( "ZVOceanExportWin.mel" );

	// menu
    pluginFn.registerUI( "CreateZelosMenu", "DeleteZelosMenu" );
	*/
	
	// Environment variables
	setenv( "BORA_RMAN_DSO_PATH", "somewhere", 0 );

    return stat;
}

MStatus uninitializePlugin( MObject obj )
{
	MStatus stat = MS::kSuccess;
	MFnPlugin pluginFn( obj );

	// Locator Nodes
	DeregisterNode( BoraBgeoImporter );
	DeregisterNode( BoraVDBFromPoints );	

    // Node data
	DeregisterData( BoraData );

	DeregisterShapeWithDrawOverride( Bora_Grid );

	/*
	// shapes
	DeregisterShapeWithDrawOverride( ZVNode	  	        );
	DeregisterShapeWithDrawOverride( ZVOceanViewer      );
	DeregisterShapeWithDrawOverride( ZVBreakingWave     );
	DeregisterShapeWithDrawOverride( ZVPtcViewer        );
	DeregisterShapeWithDrawOverride( ZVImporter         );
	DeregisterShapeWithDrawOverride( ZVExporter         );
	DeregisterShapeWithDrawOverride( ZVLauncher         );
	DeregisterShapeWithDrawOverride( ZVParticleDiffuser );
	DeregisterShapeWithDrawOverride( ZVMeshConverter    );
	DeregisterShapeWithDrawOverride( ZVOceanWave        );
	DeregisterShapeWithDrawOverride( ZVRasterizer       );
	DeregisterShapeWithDrawOverride( ZVFieldViewer      );
	DeregisterShapeWithDrawOverride( ZVRManVolArchive   );
	DeregisterShapeWithDrawOverride( ZVRManPtcArchive   );
	DeregisterShapeWithDrawOverride( ZVRManMeshArchive  );
	DeregisterShapeWithDrawOverride( ZVOceanBlend       );
	DeregisterShapeWithDrawOverride( ZVImage            );
	DeregisterShapeWithDrawOverride( ZVCacheViewer      );
	DeregisterShapeWithDrawOverride( Bora_Grid             );
	DeregisterShapeWithDrawOverride( ZVVolumeDiffuser   );
	DeregisterShapeWithDrawOverride( ZVVolumeViewer     );	
	*/

	// Commands
    DeregisterCommand( BoraVDBFromPointsCmd );

    return stat;
}

