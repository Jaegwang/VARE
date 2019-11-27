//---------------//
// BoraTest.cpp //
//-------------------------------------------------------//
// author: Julie Jang @ Dexter Studios                   //
// last update: 2018.04.10                               //
//-------------------------------------------------------//

#include <BoraTest.h>
#include <string>

MTypeId BoraTest::id( 0x300006 );
MString BoraTest::name( "BoraTest" );
MString	BoraTest::drawDbClassification( "drawdb/geometry/BoraTest" );
MString	BoraTest::drawRegistrantId( "BoraTestNodePlugin" );

MObject BoraTest::inTimeObj;
MObject BoraTest::outputObj;
MObject BoraTest::testTypeObj;
MObject BoraTest::advectionSchemeObj;
MObject BoraTest::inputAlembicPathObj;
MObject BoraTest::outputAlembicPathObj;
MObject BoraTest::gridResObj;
MObject BoraTest::scaleObj;
MObject BoraTest::xSliceObj;
MObject BoraTest::ySliceObj;
MObject BoraTest::zSliceObj;
MObject BoraTest::inMeshObj;

void* BoraTest::creator()
{
	return new BoraTest();
}

void BoraTest::postConstructor()
{
	MPxNode::postConstructor();

	nodeObj = thisMObject();
	nodeFn.setObject( nodeObj );
	dagNodeFn.setObject( nodeObj );
	nodeFn.setName( "BoraTestShape#" );


	_gridRes = -1;
	_testType = 0;
}

MStatus BoraTest::initialize()
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

    outputObj = nAttr.create( "output", "output", MFnNumericData::kFloat, 0.f, &s );
	nAttr.setHidden(1);
	CHECK_MSTATUS( addAttribute( outputObj ) );

	testTypeObj = eAttr.create( "testType", "testType", 0, &s );
    eAttr.addField( "Enright", 0 );
    eAttr.addField( "Zalesak", 1 );
	CHECK_MSTATUS( addAttribute( testTypeObj ) );

	advectionSchemeObj = eAttr.create( "advectionScheme", "advectionScheme", 0, &s );
    eAttr.addField( "Linear",		 0 );
    eAttr.addField( "RK2",			 1 );
    eAttr.addField( "RK3",			 2 );
    eAttr.addField( "RK4",			 3 );
    eAttr.addField( "MacCormack",	 4 );
    eAttr.addField( "BFECC",		 5 );
	CHECK_MSTATUS( addAttribute( advectionSchemeObj ) );

	inputAlembicPathObj = tAttr.create( "inputAlembicPath", "inputAlembicPath", MFnData::kString, &s );
	MAYA_CHECK_ERROR( s, MString("FAILED to create") + "an attribute: inputAlembicPath." );
    CHECK_MSTATUS( addAttribute( inputAlembicPathObj ) );

	outputAlembicPathObj = tAttr.create( "outputAlembicPath", "outputAlembicPath", MFnData::kString, &s );
	MAYA_CHECK_ERROR( s, MString("FAILED to create") + "an attribute: outputAlembicPath." );
    CHECK_MSTATUS( addAttribute( outputAlembicPathObj ) );

	gridResObj = nAttr.create( "gridRes", "gridRes", MFnNumericData::kInt, 100, &s );
	nAttr.setMin(1); nAttr.setSoftMax(100);
	CHECK_MSTATUS( addAttribute( gridResObj ) );

	scaleObj = nAttr.create( "scale", "scale", MFnNumericData::kFloat, 0.01f, &s );
	nAttr.setMin(0.f); nAttr.setSoftMax(10.f);
	CHECK_MSTATUS( addAttribute( scaleObj ) );

	xSliceObj = nAttr.create( "xSlice", "xSlice", MFnNumericData::kFloat, 0.f, &s );
	nAttr.setMin(0.f); nAttr.setMax(1.f);
	CHECK_MSTATUS( addAttribute( xSliceObj ) );

	ySliceObj = nAttr.create( "ySlice", "ySlice", MFnNumericData::kFloat, 0.f, &s );
	nAttr.setMin(0.f); nAttr.setMax(1.f);
	CHECK_MSTATUS( addAttribute( ySliceObj ) );

	zSliceObj = nAttr.create( "zSlice", "zSlice", MFnNumericData::kFloat, 0.f, &s );
	nAttr.setMin(0.f); nAttr.setMax(1.f);
	CHECK_MSTATUS( addAttribute( zSliceObj ) );

	inMeshObj = tAttr.create( "inMesh", "inMesh", MFnData::kMesh, &s );
	MAYA_CHECK_ERROR( s, MString("FAILED to create") + "an attribute: inMesh." );
	tAttr.setHidden(0); tAttr.setKeyable(1);
    CHECK_MSTATUS( addAttribute( inMeshObj ) );

    attributeAffects( inTimeObj,                outputObj );
    //attributeAffects( outputObj,                outputObj );
    attributeAffects( testTypeObj,              outputObj );
    attributeAffects( advectionSchemeObj,       outputObj );
    attributeAffects( inputAlembicPathObj,      outputObj );
    //attributeAffects( outputAlembicPathObj,     outputObj );
    attributeAffects( gridResObj,               outputObj );
    attributeAffects( inMeshObj,	  			outputObj );

	return MS::kSuccess;
}

MStatus BoraTest::compute( const MPlug& plug, MDataBlock& block )
{
	if( plug != outputObj ) { return MS::kUnknownParameter; }

	blockPtr = &block;
	nodeName = nodeFn.name();
	MThreadUtils::syncNumOpenMPThreads();

	const float currentTime 				= (float)block.inputValue( inTimeObj ).asTime().as( MTime::kSeconds );
	const short testType 					= block.inputValue( testTypeObj ).asShort();
	const AdvectionScheme advectionScheme 	= (AdvectionScheme)block.inputValue( advectionSchemeObj ).asShort();
	const int   gridRes  					= block.inputValue( gridResObj ) .asInt();
		  std::string inputAlembicPath		= block.inputValue( inputAlembicPathObj ).asString().asChar();
	const MString outputAlembicPath			= block.inputValue( outputAlembicPathObj ).asString();

	const bool testTypeChanged			= !(_testType == testType);
	const bool advectionSchemeChanged 	= !(_advectionScheme == advectionScheme);
	const bool gridResChanged  			= !(_gridRes == gridRes);
	const bool inputAlembicPathChanged  = !(_inputAlembicPath == inputAlembicPath );


	string p = outputAlembicPath.asChar();

	if( testTypeChanged || advectionSchemeChanged || gridResChanged || inputAlembicPathChanged )
	{
		if( testType == 0 )
		{
			_enrightTest.initialize( gridRes );
			_enrightTest.setAdvectionConditions( 1.0, 1, 10, 1.f/24.f, advectionScheme );
		}
		if( testType == 1 )
		{
//			TriangleMesh triMesh;
//			MObject meshObj = block.inputValue( inMeshObj ).asMeshTransformed();
//			Convert( triMesh, meshObj );
//			triMesh.updateBoundingBox();
			_zalesakTest.initialize( gridRes );
			_zalesakTest.setAdvectionConditions( 1.0, 1, 10, 1.f/24.f, advectionScheme );
		}
		_gridRes  		  = gridRes;
		_testType		  = testType;
		_advectionScheme  = advectionScheme;
		_inputAlembicPath = inputAlembicPath;
	}
	else
	{
		if( testType == 0 ) { _enrightTest.update( currentTime );}
		if( testType == 1 ) { _zalesakTest.update( currentTime ); }
	}

//	MMatrix m = block.inputValue( inXFormObj ).asMatrix();

    MDataHandle outputHnd = block.outputValue( outputObj );
    outputHnd.set( 0.f );

//	block.outputValue( outputObj ).set( 0.f );
	block.setClean( plug );

	toUpdateAE = true;

	return MS::kSuccess;
}

void BoraTest::draw( M3dView& view, const MDagPath& path, M3dView::DisplayStyle style, M3dView::DisplayStatus status )
{
	view.beginGL();
	{
		draw();
	}
	view.endGL();
}

void BoraTest::draw( int drawingMode )
{
    BoraTest::autoConnect();

	float scale = MPlug( nodeObj, scaleObj ).asFloat();

	float xSlice = MPlug( nodeObj, xSliceObj ).asFloat();
	float ySlice = MPlug( nodeObj, ySliceObj ).asFloat();
	float zSlice = MPlug( nodeObj, zSliceObj ).asFloat();

	Vec3f sliceRatio = Vec3f( xSlice, ySlice, zSlice );

	glPushAttrib( GL_ALL_ATTRIB_BITS );
    {
		if( _testType == 0 )
		{
			_enrightTest.drawVelocityField( true, scale, sliceRatio );
			_enrightTest.drawScalarField();
		}
		if( _testType == 1 )
		{
			_zalesakTest.drawVelocityField( true, scale, sliceRatio );
			_zalesakTest.drawScalarField();
		}
    }
	glPopAttrib();

    if( toUpdateAE )
    {
        MGlobal::executeCommand( MString( "updateAE " ) + nodeName );
        toUpdateAE = false;
    }
}

void BoraTest::autoConnect()
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

MBoundingBox BoraTest::boundingBox() const
{
	MBoundingBox bBox;

//    const AABB3f aabb = oceanTile.boundingBox();
//
//    bBox.expand( AsMPoint( aabb.minPoint() ) );
//    bBox.expand( AsMPoint( aabb.maxPoint() ) );
    bBox.expand( MPoint( -10000, -10000, -10000 ) );
    bBox.expand( MPoint(  10000,  10000,  10000 ) );
    return bBox;
}

bool BoraTest::isBounded() const
{
	return true;
}

