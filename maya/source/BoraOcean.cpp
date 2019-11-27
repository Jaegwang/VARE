//---------------//
// BoraOcean.cpp //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.12.20                               //
//-------------------------------------------------------//

#include <BoraOcean.h>

MTypeId BoraOcean::id( 0x300003 );
MString BoraOcean::name( "BoraOcean" );
MString	BoraOcean::drawDbClassification( "drawdb/geometry/BoraOcean" );
MString	BoraOcean::drawRegistrantId( "BoraOceanNodePlugin" );

MObject BoraOcean::inTimeObj;
MObject BoraOcean::inMeshObj;
MObject BoraOcean::outputObj;
MObject BoraOcean::dispersionObj;
MObject BoraOcean::spectrumObj;
MObject BoraOcean::spreadingObj;
MObject BoraOcean::filterObj;
MObject BoraOcean::randomObj;
MObject BoraOcean::numThreadsObj;
MObject BoraOcean::grainSizeObj;
MObject BoraOcean::gridLevelObj;
MObject BoraOcean::physicalLengthObj;
MObject BoraOcean::sceneConvertingScaleObj;
MObject BoraOcean::gravityObj;
MObject BoraOcean::surfaceTensionObj;
MObject BoraOcean::densityObj;
MObject BoraOcean::depthObj;
MObject BoraOcean::windDirectionObj;
MObject BoraOcean::windSpeedObj;
MObject BoraOcean::flowSpeedObj;
MObject BoraOcean::fetchObj;
MObject BoraOcean::swellObj;
MObject BoraOcean::filterSoftWidthObj;
MObject BoraOcean::filterSmallWavelengthObj;
MObject BoraOcean::filterBigWavelengthObj;
MObject BoraOcean::randomSeedObj;
MObject BoraOcean::crestGainObj;
MObject BoraOcean::crestBiasObj;
MObject BoraOcean::crestAccumulationObj;
MObject BoraOcean::crestDecayObj;
MObject BoraOcean::timeOffsetObj;
MObject BoraOcean::timeScaleObj;
MObject BoraOcean::loopingDurationObj;
MObject BoraOcean::amplitudeGainObj;
MObject BoraOcean::pinchObj;
MObject BoraOcean::mapResolutionObj;
MObject BoraOcean::displayModeObj;
MObject BoraOcean::deepWaterColorObj;
MObject BoraOcean::shallowWaterColorObj;
MObject BoraOcean::skyTextureFileObj;
MObject BoraOcean::glossinessObj;
MObject BoraOcean::exposureObj;
MObject BoraOcean::drawOutlineObj;
MObject BoraOcean::outlineColorObj;
MObject BoraOcean::outlineWidthObj;
MObject BoraOcean::showTanglesObj;
MObject BoraOcean::tangleColorObj;
MObject BoraOcean::onOffObj;

void* BoraOcean::creator()
{
	return new BoraOcean();
}

void BoraOcean::postConstructor()
{
	MPxNode::postConstructor();

	nodeObj = thisMObject();
	nodeFn.setObject( nodeObj );
	dagNodeFn.setObject( nodeObj );
	nodeFn.setName( "BoraOceanShape#" );
}

MStatus BoraOcean::initialize()
{
	MStatus s = MS::kSuccess;

	MFnEnumAttribute    eAttr;
	MFnUnitAttribute    uAttr;
	MFnTypedAttribute   tAttr;
    MFnMatrixAttribute  mAttr;
	MFnNumericAttribute nAttr;

	inTimeObj = uAttr.create( "inTime", "inTime", MFnUnitAttribute::kTime, 0.0, &s );
	uAttr.setHidden(1);
	CHECK_MSTATUS( addAttribute( inTimeObj ) );

	inMeshObj = tAttr.create( "inMesh", "inMesh", MFnData::kMesh, &s );
    tAttr.setHidden(1);
	CHECK_MSTATUS( addAttribute( inMeshObj ) );

    outputObj = nAttr.create( "output", "output", MFnNumericData::kFloat, 0.f, &s );
	nAttr.setHidden(1);
	CHECK_MSTATUS( addAttribute( outputObj ) );

	dispersionObj = eAttr.create( "dispersion", "dispersion", 0, &s );
    eAttr.addField( "Deep Water",             0 );
    eAttr.addField( "Finite Depth Water",     1 );
    eAttr.addField( "Very Small Depth Water", 2 );
	CHECK_MSTATUS( addAttribute( dispersionObj ) );

    spectrumObj = eAttr.create( "spectrum", "spectrum", 3, &s );
    eAttr.addField( "Phillips",         0 );
    eAttr.addField( "PiersonMoskowitz", 1 );
    eAttr.addField( "JOSWAP",           2 );
    eAttr.addField( "TMA",              3 );
	CHECK_MSTATUS( addAttribute( spectrumObj ) );

    spreadingObj = eAttr.create( "spreading", "spreading", 1, &s );
    eAttr.addField( "Mitsuyasu",  0 );
    eAttr.addField( "Hasselmann", 1 );
	CHECK_MSTATUS( addAttribute( spreadingObj ) );

    filterObj = eAttr.create( "filter", "filter", 0, &s );
    eAttr.addField( "None",           0 );
    eAttr.addField( "SmoothBandPass", 1 );
	CHECK_MSTATUS( addAttribute( filterObj ) );

    randomObj = eAttr.create( "random", "random", 0, &s );
    eAttr.addField( "Gaussian",  0 );
    eAttr.addField( "LogNormal", 1 );
	CHECK_MSTATUS( addAttribute( randomObj ) );

    numThreadsObj = nAttr.create( "numThreads", "numThreads", MFnNumericData::kInt, -1, &s );
    nAttr.setMin(-1); nAttr.setMax(64);
	CHECK_MSTATUS( addAttribute( numThreadsObj ) );

    grainSizeObj = nAttr.create( "grainSize", "grainSize", MFnNumericData::kInt, 512, &s );
    nAttr.setMin(32); nAttr.setMax(1024);
	CHECK_MSTATUS( addAttribute( grainSizeObj ) );

    gridLevelObj = nAttr.create( "gridLevel", "gridLevel", MFnNumericData::kInt, 9, &s );
	nAttr.setMin(5); nAttr.setMax(12);
    CHECK_MSTATUS( addAttribute( gridLevelObj ) );

    physicalLengthObj = nAttr.create( "physicalLength", "physicalLength", MFnNumericData::kFloat, 100.f, &s );
	nAttr.setMin(1.f); nAttr.setSoftMax(10000.f);
    CHECK_MSTATUS( addAttribute( physicalLengthObj ) );

    sceneConvertingScaleObj = nAttr.create( "sceneConvertingScale", "sceneConvertingScale", MFnNumericData::kFloat, 10.f, &s );
	nAttr.setMin(0.01f); nAttr.setSoftMax(100.f);
    CHECK_MSTATUS( addAttribute( sceneConvertingScaleObj ) );

    gravityObj = nAttr.create( "gravity", "gravity", MFnNumericData::kFloat, 9.81f, &s );
	nAttr.setMin(1.f); nAttr.setMax(100.f);
    CHECK_MSTATUS( addAttribute( gravityObj ) );

    surfaceTensionObj = nAttr.create( "surfaceTension", "surfaceTension", MFnNumericData::kFloat, 0.074f, &s );
	nAttr.setMin(0.001f); nAttr.setMax(1.f);
    CHECK_MSTATUS( addAttribute( surfaceTensionObj ) );

    densityObj = nAttr.create( "density", "density", MFnNumericData::kFloat, 1000.f, &s );
	nAttr.setMin(1.f); nAttr.setMax(10000.f);
    CHECK_MSTATUS( addAttribute( densityObj ) );

    depthObj = nAttr.create( "depth", "depth", MFnNumericData::kFloat, 100.f, &s );
	nAttr.setMin(0.01f); nAttr.setMax(1000.f);
    CHECK_MSTATUS( addAttribute( depthObj ) );

    windDirectionObj = nAttr.create( "windDirection", "windDirection", MFnNumericData::kFloat, 0.f, &s );
	nAttr.setMin(0.f); nAttr.setMax(360.f);
    CHECK_MSTATUS( addAttribute( windDirectionObj ) );

    windSpeedObj = nAttr.create( "windSpeed", "windSpeed", MFnNumericData::kFloat, 17.f, &s );
	nAttr.setMin(0.001f); nAttr.setSoftMax(100.f);
    CHECK_MSTATUS( addAttribute( windSpeedObj ) );

    flowSpeedObj = nAttr.create( "flowSpeed", "flowSpeed", MFnNumericData::kFloat, 0.f, &s );
	nAttr.setSoftMin(-10.f); nAttr.setSoftMax(10.f);
    CHECK_MSTATUS( addAttribute( flowSpeedObj ) );

    fetchObj = nAttr.create( "fetch", "fetch", MFnNumericData::kFloat, 300.f, &s );
	nAttr.setMin(1.f); nAttr.setMax(10000.f);
    CHECK_MSTATUS( addAttribute( fetchObj ) );

    swellObj = nAttr.create( "swell", "swell", MFnNumericData::kFloat, 0.f, &s );
	nAttr.setMin(0.f); nAttr.setMax(1.f);
    CHECK_MSTATUS( addAttribute( swellObj ) );

    filterSoftWidthObj = nAttr.create( "filterSoftWidth", "filterSoftWidth", MFnNumericData::kFloat, 1.f, &s );
	nAttr.setMin(0.f); nAttr.setSoftMax(100.f);
    CHECK_MSTATUS( addAttribute( filterSoftWidthObj ) );

    filterSmallWavelengthObj = nAttr.create( "filterSmallWavelength", "filterSmallWavelength", MFnNumericData::kFloat, 0.f, &s );
	nAttr.setMin(0.f); nAttr.setSoftMax(100.f);
    CHECK_MSTATUS( addAttribute( filterSmallWavelengthObj ) );

    filterBigWavelengthObj = nAttr.create( "filterBigWavelength", "filterBigWavelength", MFnNumericData::kFloat, 1000000.f, &s );
	nAttr.setMin(0.f); nAttr.setSoftMax(1000000.f);
    CHECK_MSTATUS( addAttribute( filterBigWavelengthObj ) );

    randomSeedObj = nAttr.create( "randomSeed", "randomSeed", MFnNumericData::kInt, 0, &s );
    nAttr.setMin(0); nAttr.setMax(1000);
    CHECK_MSTATUS( addAttribute( randomSeedObj ) );

    timeOffsetObj = nAttr.create( "timeOffset", "timeOffset", MFnNumericData::kFloat, 0.f, &s );
    nAttr.setSoftMin(0.f); nAttr.setSoftMax(10.f);
    CHECK_MSTATUS( addAttribute( timeOffsetObj ) );

    timeScaleObj = nAttr.create( "timeScale", "timeScale", MFnNumericData::kFloat, 1.f, &s );
    nAttr.setMin(0.f); nAttr.setSoftMax(100.f);
    CHECK_MSTATUS( addAttribute( timeScaleObj ) );

    loopingDurationObj = nAttr.create( "loopingDuration", "loopingDuration", MFnNumericData::kInt, 100, &s );
    nAttr.setMin(0); nAttr.setSoftMax(300);
    CHECK_MSTATUS( addAttribute( loopingDurationObj ) );

    amplitudeGainObj = nAttr.create( "amplitudeGain", "amplitudeGain", MFnNumericData::kFloat, 1.f, &s );
    nAttr.setMin(0.f); nAttr.setSoftMax(2.f);
    CHECK_MSTATUS( addAttribute( amplitudeGainObj ) );

    pinchObj = nAttr.create( "pinch", "pinch", MFnNumericData::kFloat, 0.75f, &s );
    nAttr.setMin(0.f); nAttr.setSoftMax(2.f);
    CHECK_MSTATUS( addAttribute( pinchObj ) );

    crestGainObj = nAttr.create( "crestGain", "crestGain", MFnNumericData::kFloat, 4.f, &s );
    nAttr.setMin(0.f); nAttr.setSoftMax(10.f);
    CHECK_MSTATUS( addAttribute( crestGainObj ) );

    crestBiasObj = nAttr.create( "crestBias", "crestBias", MFnNumericData::kFloat, 3.f, &s );
    nAttr.setMin(0.f); nAttr.setSoftMax(10.f);
    CHECK_MSTATUS( addAttribute( crestBiasObj ) );

    crestAccumulationObj = nAttr.create( "crestAccumulation", "crestAccumulation", MFnNumericData::kBoolean, false, &s );
    CHECK_MSTATUS( addAttribute( crestAccumulationObj ) );

    crestDecayObj = nAttr.create( "crestDecay", "crestDecay", MFnNumericData::kFloat, 0.1f, &s );
    nAttr.setMin(0.f); nAttr.setMax(1.f);
    CHECK_MSTATUS( addAttribute( crestDecayObj ) );

    mapResolutionObj = nAttr.create( "mapResolution", "mapResolution", MFnNumericData::kInt, 0 );
	nAttr.setWritable(0); nAttr.setStorable(0);
    CHECK_MSTATUS( addAttribute( mapResolutionObj ) );

	displayModeObj = eAttr.create( "displayMode", "displayMode", 2, &s );
    eAttr.addField( "None",           0 );
    eAttr.addField( "Wireframe",      1 );
    eAttr.addField( "Shaded Surface", 2 );
	CHECK_MSTATUS( addAttribute( displayModeObj ) );

    deepWaterColorObj = nAttr.createColor( "deepWaterColor", "deepWaterColor", &s );
    nAttr.setDefault(0.000f,0.001f,0.010f);
    CHECK_MSTATUS( addAttribute( deepWaterColorObj ) );

    shallowWaterColorObj = nAttr.createColor( "shallowWaterColor", "shallowWaterColor", &s );
    nAttr.setDefault(0.020f,0.140f,0.380f);
    CHECK_MSTATUS( addAttribute( shallowWaterColorObj ) );

    skyTextureFileObj = tAttr.create( "skyTextureFile", "skyTextureFile", MFnData::kString );
	tAttr.setUsedAsFilename(1);
    CHECK_MSTATUS( addAttribute( skyTextureFileObj ) );

    glossinessObj = nAttr.create( "glossiness", "glossiness", MFnNumericData::kFloat, 0.1f, &s );
    nAttr.setMin(0.f); nAttr.setMax(1.f);
    CHECK_MSTATUS( addAttribute( glossinessObj ) );

    exposureObj = nAttr.create( "exposure", "exposure", MFnNumericData::kFloat, 1.f, &s );
    nAttr.setMin(0.f); nAttr.setSoftMax(10.f);
    CHECK_MSTATUS( addAttribute( exposureObj ) );

    drawOutlineObj = nAttr.create( "drawOutline", "drawOutline", MFnNumericData::kBoolean, false, &s );
    CHECK_MSTATUS( addAttribute( drawOutlineObj ) );

    outlineColorObj = nAttr.createColor( "outlineColor", "outlineColor", &s );
    nAttr.setDefault(1.f,0.8f,0.2f);
    CHECK_MSTATUS( addAttribute( outlineColorObj ) );

    outlineWidthObj = nAttr.create( "outlineWidth", "outlineWidth", MFnNumericData::kInt, 5.f, &s );
    nAttr.setMin(1); nAttr.setMax(10);
    CHECK_MSTATUS( addAttribute( outlineWidthObj ) );

    showTanglesObj = nAttr.create( "showTangles", "showTangles", MFnNumericData::kBoolean, false, &s );
    CHECK_MSTATUS( addAttribute( showTanglesObj ) );

    tangleColorObj = nAttr.createColor( "tangleColor", "tangleColor", &s );
    nAttr.setDefault(1.f,0.f,0.f);
    CHECK_MSTATUS( addAttribute( tangleColorObj ) );

    onOffObj = nAttr.create( "onOff", "onOff", MFnNumericData::kBoolean, true, &s );
    nAttr.setHidden(1); nAttr.setStorable(0);
    CHECK_MSTATUS( addAttribute( onOffObj ) );

    attributeAffects( inTimeObj,                outputObj );
    attributeAffects( inMeshObj,                outputObj );
    attributeAffects( outputObj,                outputObj );
    attributeAffects( dispersionObj,            outputObj );
    attributeAffects( spectrumObj,              outputObj );
    attributeAffects( spreadingObj,             outputObj );
    attributeAffects( filterObj,                outputObj );
    attributeAffects( randomObj,                outputObj );
    attributeAffects( numThreadsObj,            outputObj );
    attributeAffects( grainSizeObj,             outputObj );
    attributeAffects( gridLevelObj,             outputObj );
    attributeAffects( physicalLengthObj,        outputObj );
    attributeAffects( sceneConvertingScaleObj,  outputObj );
    attributeAffects( gravityObj,               outputObj );
    attributeAffects( surfaceTensionObj,        outputObj );
    attributeAffects( densityObj,               outputObj );
    attributeAffects( depthObj,                 outputObj );
    attributeAffects( windDirectionObj,         outputObj );
    attributeAffects( windSpeedObj,             outputObj );
    attributeAffects( flowSpeedObj,             outputObj );
    attributeAffects( fetchObj,                 outputObj );
    attributeAffects( swellObj,                 outputObj );
    attributeAffects( filterSoftWidthObj,       outputObj );
    attributeAffects( filterSmallWavelengthObj, outputObj );
    attributeAffects( filterBigWavelengthObj,   outputObj );
    attributeAffects( randomSeedObj,            outputObj );
    attributeAffects( crestGainObj,             outputObj );
    attributeAffects( crestBiasObj,             outputObj );
    attributeAffects( crestAccumulationObj,     outputObj );
    attributeAffects( crestDecayObj,            outputObj );
    attributeAffects( timeOffsetObj,            outputObj );
    attributeAffects( timeScaleObj,             outputObj );
    attributeAffects( loopingDurationObj,       outputObj );
    attributeAffects( amplitudeGainObj,         outputObj );
    attributeAffects( pinchObj,                 outputObj );
    attributeAffects( onOffObj,                 outputObj );

	return MS::kSuccess;
}

MStatus BoraOcean::compute( const MPlug& plug, MDataBlock& block )
{
	if( plug != outputObj ) { return MS::kUnknownParameter; }

	blockPtr = &block;
	nodeName = nodeFn.name();
	MThreadUtils::syncNumOpenMPThreads();

	const float time = (float)block.inputValue( inTimeObj ).asTime().as( MTime::kSeconds );

	MObject meshObj = block.inputValue( inMeshObj ).asMeshTransformed();

    const bool onOff = block.inputValue( onOffObj ).asBool();

    if( onOff )
    {
        OceanParams oceanParams;
        {
            const short dispersion = block.inputValue( dispersionObj ).asShort();
            if( dispersion == 0 ) { oceanParams.dispersionRelationshipType = OceanDispersionRelationshipType::kDeepWater;           }
            if( dispersion == 1 ) { oceanParams.dispersionRelationshipType = OceanDispersionRelationshipType::kFiniteDepthWater;    }
            if( dispersion == 2 ) { oceanParams.dispersionRelationshipType = OceanDispersionRelationshipType::kVerySmallDepthWater; }

            const short spectrum = block.inputValue( spectrumObj ).asShort();
            if( spectrum == 0 ) { oceanParams.spectrumType = OceanSpectrumType::kPhillips;         }
            if( spectrum == 1 ) { oceanParams.spectrumType = OceanSpectrumType::kPiersonMoskowitz; }
            if( spectrum == 2 ) { oceanParams.spectrumType = OceanSpectrumType::kJONSWAP;          }
            if( spectrum == 3 ) { oceanParams.spectrumType = OceanSpectrumType::kTMA;              }

            const short spreading = block.inputValue( spreadingObj ).asShort();
            if( spreading == 0 ) { oceanParams.directionalSpreadingType = OceanDirectionalSpreadingType::kMitsuyasu;  }
            if( spreading == 1 ) { oceanParams.directionalSpreadingType = OceanDirectionalSpreadingType::kHasselmann; }

            const short filter = block.inputValue( filterObj ).asShort();
            if( filter == 0 ) { oceanParams.filterType = OceanFilterType::kNoFilter;       }
            if( filter == 1 ) { oceanParams.filterType = OceanFilterType::kSmoothBandPass; }

            const short random = block.inputValue( randomObj ).asShort();
            if( random == 0 ) { oceanParams.randomType = OceanRandomType::kGaussian;  }
            if( random == 1 ) { oceanParams.randomType = OceanRandomType::kLogNormal; }

            oceanParams.numThreads            = block.inputValue( numThreadsObj            ).asInt();
            oceanParams.grainSize             = block.inputValue( grainSizeObj             ).asInt();
            oceanParams.gridLevel             = block.inputValue( gridLevelObj             ).asInt();
            oceanParams.physicalLength        = block.inputValue( physicalLengthObj        ).asFloat();
            oceanParams.sceneConvertingScale  = block.inputValue( sceneConvertingScaleObj  ).asFloat();
            oceanParams.gravity               = block.inputValue( gravityObj               ).asFloat();
            oceanParams.surfaceTension        = block.inputValue( surfaceTensionObj        ).asFloat();
            oceanParams.density               = block.inputValue( densityObj               ).asFloat();
            oceanParams.depth                 = block.inputValue( depthObj                 ).asFloat();
            oceanParams.windDirection         = block.inputValue( windDirectionObj         ).asFloat();
            oceanParams.windSpeed             = block.inputValue( windSpeedObj             ).asFloat();
            oceanParams.flowSpeed             = block.inputValue( flowSpeedObj             ).asFloat();
            oceanParams.fetch                 = block.inputValue( fetchObj                 ).asFloat();
            oceanParams.swell                 = block.inputValue( swellObj                 ).asFloat();
            oceanParams.filterSoftWidth       = block.inputValue( filterSoftWidthObj       ).asFloat();
            oceanParams.filterSmallWavelength = block.inputValue( filterSmallWavelengthObj ).asFloat();
            oceanParams.filterBigWavelength   = block.inputValue( filterBigWavelengthObj   ).asFloat();
            oceanParams.randomSeed            = block.inputValue( randomSeedObj            ).asInt();
            oceanParams.crestGain             = block.inputValue( crestGainObj             ).asFloat();
            oceanParams.crestBias             = block.inputValue( crestBiasObj             ).asFloat();
            oceanParams.crestAccumulation     = block.inputValue( crestAccumulationObj     ).asBool();
            oceanParams.crestDecay            = block.inputValue( crestDecayObj            ).asFloat();
            oceanParams.timeOffset            = block.inputValue( timeOffsetObj            ).asFloat();
            oceanParams.timeScale             = block.inputValue( timeScaleObj             ).asFloat();
            oceanParams.loopingDuration       = block.inputValue( loopingDurationObj       ).asInt() * ( 1.f / CurrentFPS() );
            oceanParams.amplitudeGain         = block.inputValue( amplitudeGainObj         ).asFloat();
            oceanParams.pinch                 = block.inputValue( pinchObj                 ).asFloat();
        }

        if( oceanTile.initialize( oceanParams ) )
        {
            oceanTileVertexData.initialize( oceanTile );
        }

        BoraOcean::calcTileRange();

        oceanTile.update( time );
        oceanTileVertexData.update( time );

        if( _gridLevel != oceanParams.gridLevel )
        {
            block.outputValue( mapResolutionObj ).set( oceanParams.resolution() );
            _gridLevel = oceanParams.gridLevel;
            toUpdateAE = true;
        }
    }

	block.outputValue( outputObj ).set( 0.f );
	block.setClean( plug );

	return MS::kSuccess;
}

void BoraOcean::calcTileRange()
{
    MDataBlock& block = *blockPtr;

    MObject meshObj = block.inputValue( inMeshObj ).asMeshTransformed();

    if( meshObj.isNull() )
    {
        i0 = j0 = 0;
        i1 = j1 = 1;

        const float L = oceanTileVertexData.tileSize();

        aabb.reset();
        aabb.expand( Vec3f(   L, 0.f, 0.f ) );
        aabb.expand( Vec3f( 0.f, 0.f,   L ) );
        aabb.expand();
    }
    else
    {
        const float L = oceanTileVertexData.tileSize();

        Convert( inputMesh, meshObj, false, "currentUVSet" );

        aabb = inputMesh.boundingBox();
        aabb.expand();

        const Vec3f& minPt = aabb.minPoint();
        const Vec3f& maxPt = aabb.maxPoint();

        i0 = std::floor( minPt.x / L );
        i1 = std::floor( maxPt.x / L ) + 1;

        j0 = std::floor( minPt.z / L );
        j1 = std::floor( maxPt.z / L ) + 1;
    }
}

void BoraOcean::draw( M3dView& view, const MDagPath& path, M3dView::DisplayStyle style, M3dView::DisplayStatus status )
{
	view.beginGL();
	{
		draw();
	}
	view.endGL();
}

void BoraOcean::draw( int drawingMode )
{
    BoraOcean::autoConnect();

    if( oceanTileVertexData.initialized() == false )
    {
        return;
    }

    Vec3f cameraPosition;
    {
        MDagPath camDagPath;
        M3dView view = M3dView::active3dView();
        view.getCamera( camDagPath );
        MFnCamera camFn( camDagPath );

        MPoint E = camFn.eyePoint( MSpace::kWorld );
        cameraPosition = Vec3f( (float)E.x, (float)E.z, (float)E.y );
    }

    short displayMode;
    {
        MPlug plg( nodeObj, displayModeObj );
        displayMode = plg.asShort();
    }

    Vec3f deepWaterColor;
    {
        MPlug plg( nodeObj, deepWaterColorObj );
        plg.child(0).getValue( deepWaterColor.r );
        plg.child(1).getValue( deepWaterColor.g );
        plg.child(2).getValue( deepWaterColor.b );
    }

    Vec3f shallowWaterColor;
    {
        MPlug plg( nodeObj, shallowWaterColorObj );
        plg.child(0).getValue( shallowWaterColor.r );
        plg.child(1).getValue( shallowWaterColor.g );
        plg.child(2).getValue( shallowWaterColor.b );
    }

    MString skyTextureFile;
    {
        MPlug plg( nodeObj, skyTextureFileObj );
        skyTextureFile = plg.asString();
    }

    float glossiness;
    {
        MPlug plg( nodeObj, glossinessObj );
        glossiness = plg.asFloat();
    }

    float exposure;
    {
        MPlug plg( nodeObj, exposureObj );
        exposure = plg.asFloat();
    }

    bool drawOutline;
    {
        MPlug plg( nodeObj, drawOutlineObj );
        drawOutline = plg.asBool();
    }

    Vec3f outlineColor;
    {
        MPlug plg( nodeObj, outlineColorObj );
        plg.child(0).getValue( outlineColor.r );
        plg.child(1).getValue( outlineColor.g );
        plg.child(2).getValue( outlineColor.b );
    }

    float outlineWidth;
    {
        MPlug plg( nodeObj, outlineWidthObj );
        outlineWidth = plg.asInt();
    }

    int showTangles;
    {
        MPlug plg( nodeObj, showTanglesObj );
        showTangles = (int)plg.asBool();
    }

    Vec3f tangleColor;
    {
        MPlug plg( nodeObj, tangleColorObj );
        plg.child(0).getValue( tangleColor.r );
        plg.child(1).getValue( tangleColor.g );
        plg.child(2).getValue( tangleColor.b );
    }

    const float L = oceanTileVertexData.tileSize();

    glPushAttrib( GL_ALL_ATTRIB_BITS );
    {
        for( int i=i0; i<i1; ++i )
        {
            for( int j=j0; j<j1; ++j )
            {
                glPushMatrix();
                {
                    glTranslate( i*L, 0.f, j*L );

                    if( displayMode == 0 )
                    {
                        continue;
                    }
                    else if( displayMode == 1 ) // wireframe
                    {
                        glLineWidth( 1 );
                        glColor3f( 1, 1, 1 );
                        oceanTileVertexData.drawWireframe( false, false );
                    }
                    else
                    {
                        glDrawOcean.draw
                        (
                            oceanTileVertexData,
                            cameraPosition,
                            deepWaterColor,
                            shallowWaterColor,
                            skyTextureFile.asChar(),
                            glossiness,
                            exposure,
                            showTangles,
                            tangleColor
                        );
                    }

                    if( drawOutline )
                    {
                        glColor( outlineColor );
                        glLineWidth( outlineWidth );
                        oceanTileVertexData.drawOutline();
                    }
                }
                glPopMatrix();
            }
        }

        glLineWidth(1);
    }
    glPopAttrib();

    if( toUpdateAE )
    {
        MGlobal::executeCommand( MString( "updateAE " ) + nodeName );
        toUpdateAE = false;
    }
}

void BoraOcean::autoConnect()
{
    if( !isThe1stTimeDraw ) { return; }

    isThe1stTimeDraw = false;

    MDGModifier mod;

    // time1.outTime -> BoraOceanShape#.inTime
    {
        MObject time1NodeObj = NodeNameToMObject( "time1" );
        MFnDependencyNode time1NodeFn( time1NodeObj );

        MPlug fromPlg = time1NodeFn.findPlug( "outTime" );
        MPlug toPlg   = MPlug( nodeObj, inTimeObj );

        if( !toPlg.isConnected() )
        {
            mod.connect( fromPlg, toPlg );
        }
    }

    // BoraOceanShape#.output -> BoraOcean#.dynamics
    {
        MObject parentObj = dagNodeFn.parent( 0 );
        MFnDependencyNode parentXFormDGFn( parentObj );

        MPlug fromPlg = MPlug( nodeObj, outputObj );
        MPlug toPlg   = parentXFormDGFn.findPlug( "dynamics" );

        if( !fromPlg.isConnected() )
        {
            mod.connect( fromPlg, toPlg );
        }
    }

    mod.doIt();
}

MBoundingBox BoraOcean::boundingBox() const
{
	MBoundingBox bBox;

    bBox.expand( AsMPoint( aabb.minPoint() ) );
    bBox.expand( AsMPoint( aabb.maxPoint() ) );

    return bBox;
}

bool BoraOcean::isBounded() const
{
	return true;
}

const OceanParams& BoraOcean::getOceanParams() const
{
    return oceanTile.oceanParams();
}

const OceanTile& BoraOcean::getOceanTile() const
{
    return oceanTile;
}

const OceanTileVertexData& BoraOcean::getOceanTileVertexData() const
{
    return oceanTileVertexData;
}

