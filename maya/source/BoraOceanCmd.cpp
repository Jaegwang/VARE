//------------------//
// BoraOceanCmd.cpp //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.04.06                               //
//-------------------------------------------------------//

#include <BoraOceanCmd.h>

MString BoraOceanCmd::name( "BoraOceanCmd" );

#define nodeNameFlag       "-nN"
#define nodeNameLongFlag   "-nodeName"
#define toolNameFlag       "-tN"
#define toolNameLongFlag   "-toolName"
#define filePathFlag       "-fP"
#define filePathLongFlag   "-filePath"
#define fileNameFlag       "-fN"
#define fileNameLongFlag   "-fileName"
#define startFrameFlag     "-sF"
#define startFrameLongFlag "-startFrame"
#define endFrameFlag        "-eF"
#define endFrameLongFlag    "-endFrame"

MSyntax
BoraOceanCmd::newSyntax()
{
	MSyntax syntax;

	syntax.addFlag( nodeNameFlag,   nodeNameLongFlag,   MSyntax::kString   );
	syntax.addFlag( toolNameFlag,   toolNameLongFlag,   MSyntax::kString   );
	syntax.addFlag( filePathFlag,   filePathLongFlag,   MSyntax::kString   );
	syntax.addFlag( fileNameFlag,   fileNameLongFlag,   MSyntax::kString   );
	syntax.addFlag( startFrameFlag, startFrameLongFlag, MSyntax::kUnsigned );
	syntax.addFlag( endFrameFlag,   endFrameLongFlag,   MSyntax::kUnsigned );

	return syntax;
}

MStatus
BoraOceanCmd::doIt( const MArgList& args )
{
	MStatus stat = MS::kSuccess;

	MArgDatabase argData( syntax(), args, &stat );
	if( !stat ) { return MS::kFailure; }

	MString nodeName   = getNodeName   ( argData );
	MString toolName   = getToolName   ( argData );
	MString filePath   = getFilePath   ( argData );
	MString fileName   = getFileName   ( argData );
    int     startFrame = getStartFrame ( argData );
    int     endFrame   = getEndFrame   ( argData );

    if( toolName == "export" )
    {
        BoraOceanCmd::exportData( nodeName, filePath, fileName, startFrame, endFrame );
    }
    else if( toolName == "import" )
    {
        BoraOceanCmd::importData( nodeName, filePath, fileName );
    }
    else
    {
        MGlobal::displayError( "Invalid tool name" );
        return MS::kFailure;
    }

	return MS::kSuccess;
}

MString BoraOceanCmd::getNodeName( const MArgDatabase& argData )
{
	MString nodeName;

	if( argData.isFlagSet( nodeNameFlag ) )
	{
		if( !argData.getFlagArgument( nodeNameFlag, 0, nodeName ) )
		{
			MGlobal::displayError( name + ": No -nodeName flag." );
			return nodeName;
		}
	}

	return nodeName;
}

MString BoraOceanCmd::getToolName( const MArgDatabase& argData )
{
	MString toolName;

	if( argData.isFlagSet( toolNameFlag ) )
	{
		if( !argData.getFlagArgument( toolNameFlag, 0, toolName ) )
		{
			MGlobal::displayError( name + ": No -toolName flag." );
			return toolName;
		}
	}

	return toolName;
}

MString BoraOceanCmd::getFilePath( const MArgDatabase& argData )
{
	MString filePath;

	if( argData.isFlagSet( filePathFlag ) )
	{
		if( !argData.getFlagArgument( filePathFlag, 0, filePath ) )
		{
			MGlobal::displayError( name + ": No -filePath flag." );
			return filePath;
		}
	}

	return filePath;
}

MString BoraOceanCmd::getFileName( const MArgDatabase& argData )
{
	MString fileName;

	if( argData.isFlagSet( fileNameFlag ) )
	{
		if( !argData.getFlagArgument( fileNameFlag, 0, fileName ) )
		{
			MGlobal::displayError( name + ": No -fileName flag." );
			return fileName;
		}
	}

	return fileName;
}

int BoraOceanCmd::getStartFrame( const MArgDatabase& argData )
{
    unsigned int startFrame = 0;

    if( argData.isFlagSet( startFrameFlag ) )
    {
        if( !argData.getFlagArgument( startFrameFlag, 0, startFrame ) )
        {
            return 0; // default startFrameue
        }
    }

    return (int)startFrame;
}

int BoraOceanCmd::getEndFrame( const MArgDatabase& argData )
{
    unsigned int endFrame = 0;

    if( argData.isFlagSet( endFrameFlag ) )
    {
        if( !argData.getFlagArgument( endFrameFlag, 0, endFrame ) )
        {
            return 0; // default endFrameue
        }
    }

    return (int)endFrame;
}

void BoraOceanCmd::exportData( MString nodeName, MString filePath, MString fileName, int startFrame, int endFrame )
{
	MObject BoraOceanNodeObj = NodeNameToMObject( nodeName );

    if( BoraOceanNodeObj.isNull() )
    {
        MGlobal::displayError( "Invalid node name" );
        return;
    }

    MFnDependencyNode BoraOceanNodeFn( BoraOceanNodeObj );
    const BoraOcean* BoraOceanNodePtr = (BoraOcean*)BoraOceanNodeFn.userNode();
    const BoraOcean& BoraOceanNode = *BoraOceanNodePtr;

    const OceanParams& oceanParams = BoraOceanNode.getOceanParams();
    const OceanTileVertexData& oceanTileVertexData = BoraOceanNode.getOceanTileVertexData();

    MString filePathName = filePath + "/" + fileName + ".oceanParams";

    if( oceanParams.save( filePath.asChar(), fileName.asChar() ) == false )
    {
        MGlobal::displayError( "Failed to export" );
        return;
    }

    MComputation computation;
    computation.beginComputation();
    {
        for( int frame=startFrame; frame<=endFrame; ++frame )
        {
            if( computation.isInterruptRequested() ) { break; }

            MGlobal::viewFrame( frame );

            const std::string pad = MakePadding( frame, 4 );

            filePathName = filePath + "/" + fileName + "." + pad.c_str() + ".exr";

            oceanTileVertexData.exportToEXR( filePathName.asChar() );
        }
    }
    computation.endComputation();
}

void BoraOceanCmd::importData( MString nodeName, MString filePath, MString fileName )
{
	MObject BoraOceanNodeObj = NodeNameToMObject( nodeName );

    if( BoraOceanNodeObj.isNull() )
    {
        MGlobal::displayError( "Invalid node name" );
        return;
    }

    MFnDependencyNode BoraOceanNodeFn( BoraOceanNodeObj );
    const BoraOcean* BoraOceanNodePtr = (BoraOcean*)BoraOceanNodeFn.userNode();
    const BoraOcean& BoraOceanNode = *BoraOceanNodePtr;

    const MString filePathName = filePath + "/" + fileName + ".oceanParams";

    if( DoesFileExist( filePathName.asChar() ) == false )
    {
        MGlobal::displayInfo( "No file" );
        return;
    }

    OceanParams oceanParams;

    if( oceanParams.load( filePath.asChar(), fileName.asChar() ) == false )
    {
        MGlobal::displayError( "Failed to import" );
        return;
    }

    MString cmd;
    {
        int   iValue = 0;
        float fValue = 0.f;

        iValue = 0;
        cmd += MString("setAttr ") + nodeName + "." + "onOff " + iValue + ";";

        iValue = (int)oceanParams.dispersionRelationshipType;
        cmd += MString("setAttr ") + nodeName + "." + "dispersion " + iValue + ";";

        iValue = (int)oceanParams.spectrumType;
        cmd += MString("setAttr ") + nodeName + "." + "spectrum " + iValue + ";";

        iValue = (int)oceanParams.directionalSpreadingType;
        cmd += MString("setAttr ") + nodeName + "." + "spreading " + iValue + ";";

        iValue = (int)oceanParams.filterType;
        cmd += MString("setAttr ") + nodeName + "." + "filter " + iValue + ";";

        iValue = (int)oceanParams.randomType;
        cmd += MString("setAttr ") + nodeName + "." + "random " + iValue + ";";

        iValue = oceanParams.numThreads;
        cmd += MString("setAttr ") + nodeName + "." + "numThreads " + iValue + ";";

        iValue = oceanParams.grainSize;
        cmd += MString("setAttr ") + nodeName + "." + "grainSize " + iValue + ";";

        iValue = oceanParams.gridLevel;
        cmd += MString("setAttr ") + nodeName + "." + "gridLevel " + iValue + ";";

        fValue = oceanParams.physicalLength;
        cmd += MString("setAttr ") + nodeName + "." + "physicalLength " + fValue + ";";

        fValue = oceanParams.sceneConvertingScale;
        cmd += MString("setAttr ") + nodeName + "." + "sceneConvertingScale " + fValue + ";";

        fValue = oceanParams.gravity;
        cmd += MString("setAttr ") + nodeName + "." + "gravity " + fValue + ";";

        fValue = oceanParams.surfaceTension;
        cmd += MString("setAttr ") + nodeName + "." + "surfaceTension " + fValue + ";";

        fValue = oceanParams.density;
        cmd += MString("setAttr ") + nodeName + "." + "density " + fValue + ";";

        fValue = oceanParams.depth;
        cmd += MString("setAttr ") + nodeName + "." + "depth " + fValue + ";";

        fValue = oceanParams.windDirection;
        cmd += MString("setAttr ") + nodeName + "." + "windDirection " + fValue + ";";

        fValue = oceanParams.windSpeed;
        cmd += MString("setAttr ") + nodeName + "." + "windSpeed " + fValue + ";";

        fValue = oceanParams.fetch;
        cmd += MString("setAttr ") + nodeName + "." + "fetch " + fValue + ";";

        fValue = oceanParams.swell;
        cmd += MString("setAttr ") + nodeName + "." + "swell " + fValue + ";";

        fValue = oceanParams.filterSoftWidth;
        cmd += MString("setAttr ") + nodeName + "." + "filterSoftWidth " + fValue + ";";

        fValue = oceanParams.filterSmallWavelength;
        cmd += MString("setAttr ") + nodeName + "." + "filterSmallWavelength " + fValue + ";";

        fValue = oceanParams.filterBigWavelength;
        cmd += MString("setAttr ") + nodeName + "." + "filterBigWavelength " + fValue + ";";

        iValue = oceanParams.randomSeed;
        cmd += MString("setAttr ") + nodeName + "." + "randomSeed " + iValue + ";";

        fValue = oceanParams.timeOffset;
        cmd += MString("setAttr ") + nodeName + "." + "timeOffset " + fValue + ";";

        fValue = oceanParams.timeScale;
        cmd += MString("setAttr ") + nodeName + "." + "timeScale " + fValue + ";";

        fValue = oceanParams.loopingDuration * CurrentFPS();
        cmd += MString("setAttr ") + nodeName + "." + "loopingDuration " + fValue + ";";

        fValue = oceanParams.amplitudeGain;
        cmd += MString("setAttr ") + nodeName + "." + "amplitudeGain " + fValue + ";";

        fValue = oceanParams.pinch;
        cmd += MString("setAttr ") + nodeName + "." + "pinch " + fValue + ";";

        fValue = oceanParams.crestGain;
        cmd += MString("setAttr ") + nodeName + "." + "crestGain " + fValue + ";";

        fValue = oceanParams.crestBias;
        cmd += MString("setAttr ") + nodeName + "." + "crestBias " + fValue + ";";

        fValue = oceanParams.crestAccumulation;
        cmd += MString("setAttr ") + nodeName + "." + "crestAccumulation " + fValue + ";";

        fValue = oceanParams.crestDecay;
        cmd += MString("setAttr ") + nodeName + "." + "crestDecay " + fValue + ";";

        iValue = 1;
        cmd += MString("setAttr ") + nodeName + "." + "onOff " + iValue + ";";
    }
    MGlobal::executeCommand( cmd );
}

