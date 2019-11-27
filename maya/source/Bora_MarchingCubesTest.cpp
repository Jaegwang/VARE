//----------------------//
// Bora_MarchingCubesTest.cpp //
//-------------------------------------------------------//
// author: Julie Jang @ Dexter Studios                   //
// last update: 2018.04.10                               //
//-------------------------------------------------------//

#include <Bora_MarchingCubesTest.h>

MTypeId Bora_MarchingCubesTest::id( 0x300007 );
MString Bora_MarchingCubesTest::name( "Bora_MarchingCubesTest" );

MString	Bora_MarchingCubesTest::drawDbClassification( "drawdb/geometry/Bora_MarchingCubesTest" );
MString	Bora_MarchingCubesTest::drawRegistrantId( "ZarVisForMayaPlugin" );

MObject Bora_MarchingCubesTest::inputObj;
MObject Bora_MarchingCubesTest::inXFormObj;
MObject Bora_MarchingCubesTest::outputObj;
MObject Bora_MarchingCubesTest::subdivisionObj;
MObject Bora_MarchingCubesTest::inMeshObj;

Bora_MarchingCubesTest::Bora_MarchingCubesTest()
{
	toUpdateAE = true;
}

Bora_MarchingCubesTest::~Bora_MarchingCubesTest()
{

}

void*
Bora_MarchingCubesTest::creator()
{
    return new Bora_MarchingCubesTest();
}

void
Bora_MarchingCubesTest::postConstructor()
{
    nodeObj = thisMObject();
    nodeFn.setObject( nodeObj );
    dagNodeFn.setObject( nodeObj );
    nodeFn.setName( "Bora_MarchingCubesTestShape#" );
}

MStatus
Bora_MarchingCubesTest::initialize()
{
	MStatus s = MS::kSuccess;

	MFnEnumAttribute    eAttr;
    MFnTypedAttribute   tAttr;
    MFnMatrixAttribute  mAttr;
    MFnNumericAttribute nAttr;

    inputObj = nAttr.create( "input", "input", MFnNumericData::kFloat, 0.f, &s );
	MAYA_CHECK_ERROR( s, MString("FAILED to create") + "an attribute: input." );
    CHECK_MSTATUS( addAttribute( inputObj ) );

    inXFormObj = mAttr.create( "inXForm", "inXForm", MFnMatrixAttribute::kDouble, &s );
	MAYA_CHECK_ERROR( s, MString("FAILED to create") + "an attribute: inXForm." );
	mAttr.setHidden(0);
	CHECK_MSTATUS( addAttribute( inXFormObj ) );

    outputObj = nAttr.create( "output", "output", MFnNumericData::kFloat, 0.f, &s );
	MAYA_CHECK_ERROR( s, MString("FAILED to create") + "an attribute: output." );
    CHECK_MSTATUS( addAttribute( outputObj ) );

    subdivisionObj = nAttr.create( "subdivision", "subdivision", MFnNumericData::kInt, 10, &s );
	MAYA_CHECK_ERROR( s, MString("FAILED to create") + "an attribute: subdivision." );
    nAttr.setMin(10); nAttr.setMax(1024);
	CHECK_MSTATUS( addAttribute( subdivisionObj ) );

	inMeshObj = tAttr.create( "inMesh", "inMesh", MFnData::kMesh, &s );
	MAYA_CHECK_ERROR( s, MString("FAILED to create") + "an attribute: inMesh." );
	tAttr.setHidden(0); tAttr.setKeyable(1);
    CHECK_MSTATUS( addAttribute( inMeshObj ) );

    attributeAffects( inputObj,       outputObj );
	attributeAffects( inXFormObj,     outputObj );
    attributeAffects( subdivisionObj, outputObj );
    attributeAffects( inMeshObj,	  outputObj );

    return MS::kSuccess;
}

MStatus
Bora_MarchingCubesTest::compute( const MPlug& plug, MDataBlock& block )
{
    if( plug != outputObj ) { return MS::kUnknownParameter; }

    blockPtr = &block;
    nodeName = nodeFn.name();
    MThreadUtils::syncNumOpenMPThreads();

    const int subdivision = block.inputValue( subdivisionObj ).asInt();
	MObject meshObj = block.inputValue( inMeshObj ).asMeshTransformed();
	// TODO: if no mesh, return

	_triMesh.reset();
	Convert( _triMesh, meshObj ); //defined in MayaUtils.h
	_triMesh.updateBoundingBox();
	AABB3f aabb = AABB3f( Vec3f(-50.f), Vec3f(50.f) );
	Grid grid( subdivision, aabb );
	_lvs.setGrid( grid );
	_stt.setGrid( grid );
	//_lvsSolver.reset();
	_lvsSolver.set( grid );
	_lvsSolver.addMesh( _lvs, _stt, _triMesh, BORA_LARGE, BORA_LARGE );
	_lvsSolver.finalize();

	// run marching cubes on _lvs.
	_outputTriMesh.reset();
	_marchingCubes.reset();
	_marchingCubes.compute( _lvs, _outputTriMesh );
	_outputTriMesh.updateBoundingBox();

	MMatrix m = block.inputValue( inXFormObj ).asMatrix();

    childChanged( MPxSurfaceShape::kBoundingBoxChanged );

    MDataHandle outputHnd = block.outputValue( outputObj );
    outputHnd.set( 0.f );

    block.setClean( plug );

    toUpdateAE = true;

    return MS::kSuccess;
}

MBoundingBox Bora_MarchingCubesTest::boundingBox() const
{
    //const AABB3f aabb = _densityField.boundingBox();

    MBoundingBox bBox;

//    bBox.expand( AsMPoint( aabb.minPoint() ) );
//    bBox.expand( AsMPoint( aabb.maxPoint() ) );
    bBox.expand( MPoint( -10000, -10000, -10000 ) );
    bBox.expand( MPoint(  10000,  10000,  10000 ) );

    return bBox;
}

bool Bora_MarchingCubesTest::isBounded() const
{
    return true;
}

void Bora_MarchingCubesTest::draw( int drawingMode )
{
    _autoConnect();

    const float end = MPlug( nodeObj, outputObj ).asFloat();

	if( !_outputTriMesh.numTriangles() ) { return; }

	glPushAttrib( GL_ALL_ATTRIB_BITS );
	_outputTriMesh.drawWireframe();
	glPopAttrib();


//    const float end = MPlug( nodeObj, outputObj ).asFloat();
//
//    if( toUpdateAE )
//    {
//        MGlobal::executeCommand( MString( "updateAE " ) + nodeName );
//        toUpdateAE = false;
//    }
}

void Bora_MarchingCubesTest::_autoConnect()
{
    if( !isThe1stDraw ) { return; }
    isThe1stDraw = false;

    MPlug toPlg = MPlug( nodeObj, inXFormObj );
    if( toPlg.isConnected() ) { return; }

    MDGModifier mod;

    MFnDagNode parentXFormDagFn( nodeObj );
    MObject parentObj = parentXFormDagFn.parent( 0 );
    parentXFormDagFn.setObject( parentObj );

    const MString xformName = parentXFormDagFn.name();

    MFnDependencyNode parentXFormDGFn( parentObj );
    MPlug fromPlg = parentXFormDGFn.findPlug( "matrix" );

    mod.connect( fromPlg, toPlg );

    mod.doIt();
}


