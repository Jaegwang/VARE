//--------------//
// BoraGrid.cpp //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.01.24                               //
//-------------------------------------------------------//

#include <BoraGrid.h>

MTypeId BoraGrid::id( 0x300001 );
MString BoraGrid::name( "BoraGrid" );
MString	BoraGrid::drawDbClassification( "drawdb/geometry/BoraGrid" );
MString	BoraGrid::drawRegistrantId( "BoraGridNodePlugin" );

MObject BoraGrid::inputObj;
MObject BoraGrid::inXFormObj;
MObject BoraGrid::outputObj;
MObject BoraGrid::subdivisionObj;
MObject BoraGrid::displayGridObj;
MObject BoraGrid::dispGridX0Obj;
MObject BoraGrid::dispGridX1Obj;
MObject BoraGrid::dispGridY0Obj;
MObject BoraGrid::dispGridY1Obj;
MObject BoraGrid::dispGridZ0Obj;
MObject BoraGrid::dispGridZ1Obj;
MObject BoraGrid::resolutionObj;
MObject BoraGrid::scalarFieldMemorySizeObj;
MObject BoraGrid::vectorFieldMemorySizeObj;

void* BoraGrid::creator()
{
	return new BoraGrid();
}

void BoraGrid::postConstructor()
{
	MPxNode::postConstructor();

	nodeObj = thisMObject();
	nodeFn.setObject( nodeObj );
	dagNodeFn.setObject( nodeObj );
	nodeFn.setName( "BoraGridShape#" );
}

MStatus BoraGrid::initialize()
{
	MStatus s = MS::kSuccess;

	MFnEnumAttribute    eAttr;
	MFnUnitAttribute    uAttr;
	MFnTypedAttribute   tAttr;
    MFnMatrixAttribute  mAttr;
	MFnNumericAttribute nAttr;

    inputObj = nAttr.create( "input", "input", MFnNumericData::kFloat, 0.f, &s );
    CHECK_MSTATUS( addAttribute( inputObj ) );

    inXFormObj = mAttr.create( "inXForm", "inXForm", MFnMatrixAttribute::kFloat, &s );
	mAttr.setHidden(0);
	CHECK_MSTATUS( addAttribute( inXFormObj ) );

    outputObj = nAttr.create( "output", "output", MFnNumericData::kFloat, 0.f, &s );
    CHECK_MSTATUS( addAttribute( outputObj ) );

    subdivisionObj = nAttr.create( "subdivision", "subdivision", MFnNumericData::kInt, 10, &s );
    nAttr.setMin(10); nAttr.setMax(1024);
	CHECK_MSTATUS( addAttribute( subdivisionObj ) );

    displayGridObj = nAttr.create( "displayGrid", "displayGrid", MFnNumericData::kBoolean, true, &s );
    CHECK_MSTATUS( addAttribute( displayGridObj ) );

    dispGridX0Obj = nAttr.create( "dispGridX0", "dispGridX0", MFnNumericData::kBoolean, true, &s );
    CHECK_MSTATUS( addAttribute( dispGridX0Obj ) );

    dispGridX1Obj = nAttr.create( "dispGridX1", "dispGridX1", MFnNumericData::kBoolean, false, &s );
    CHECK_MSTATUS( addAttribute( dispGridX1Obj ) );

    dispGridY0Obj = nAttr.create( "dispGridY0", "dispGridY0", MFnNumericData::kBoolean, true, &s );
    CHECK_MSTATUS( addAttribute( dispGridY0Obj ) );

    dispGridY1Obj = nAttr.create( "dispGridY1", "dispGridY1", MFnNumericData::kBoolean, false, &s );
    CHECK_MSTATUS( addAttribute( dispGridY1Obj ) );

    dispGridZ0Obj = nAttr.create( "dispGridZ0", "dispGridZ0", MFnNumericData::kBoolean, true, &s );
    CHECK_MSTATUS( addAttribute( dispGridZ0Obj ) );

    dispGridZ1Obj = nAttr.create( "dispGridZ1", "dispGridZ1", MFnNumericData::kBoolean, false, &s );
    CHECK_MSTATUS( addAttribute( dispGridZ1Obj ) );

    resolutionObj = nAttr.create( "resolution", "resolution", MFnNumericData::k3Int, 0, &s );
    nAttr.setWritable(0);
    CHECK_MSTATUS( addAttribute( resolutionObj ) );

	MFnStringData strDataFn;
	MObject defaultStrObj( strDataFn.create( "0 bytes" ) );

    scalarFieldMemorySizeObj = tAttr.create( "scalarFieldMemorySize", "scalarFieldMemorySize", MFnData::kString, defaultStrObj, &s );
    tAttr.setWritable(0);
    CHECK_MSTATUS( addAttribute( scalarFieldMemorySizeObj ) );

    vectorFieldMemorySizeObj = tAttr.create( "vectorFieldMemorySize", "vectorFieldMemorySize", MFnData::kString, defaultStrObj, &s );
    tAttr.setWritable(0);
    CHECK_MSTATUS( addAttribute( vectorFieldMemorySizeObj ) );

    attributeAffects( inputObj,       outputObj );
	attributeAffects( inXFormObj,     outputObj );
    attributeAffects( subdivisionObj, outputObj );

	return MS::kSuccess;
}

MStatus BoraGrid::compute( const MPlug& plug, MDataBlock& block )
{
	if( plug != outputObj ) { return MS::kUnknownParameter; }

	blockPtr = &block;
	nodeName = nodeFn.name();
	MThreadUtils::syncNumOpenMPThreads();

    const int subdivision = block.inputValue( subdivisionObj ).asInt();
    grid.initialize( subdivision, AABB3f( Vec3f(-0.5f), Vec3f(0.5f) ) );

    MMatrix m = block.inputValue( inXFormObj ).asMatrix();
    //grid.applyTransform( AsMat44f(m) );

    scalarFieldMemorySize = grid.memorySize( DataType::kFloat32 ).c_str();
    vectorFieldMemorySize = grid.memorySize( DataType::kVec3f   ).c_str();

    block.outputValue( resolutionObj ).set( (int)grid.nx(), (int)grid.ny(), (int)grid.nz() );
    block.outputValue( scalarFieldMemorySizeObj ).set( scalarFieldMemorySize );
    block.outputValue( vectorFieldMemorySizeObj ).set( vectorFieldMemorySize );

    MDataHandle outputHnd = block.outputValue( outputObj );
    outputHnd.set( 0.f );

    block.setClean( plug );

    toUpdateAE = true;

	return MS::kSuccess;
}

void BoraGrid::draw( M3dView& view, const MDagPath& path, M3dView::DisplayStyle style, M3dView::DisplayStatus status )
{
	view.beginGL();
	{
		draw();
	}
	view.endGL();
}

void BoraGrid::draw( int drawingMode )
{
    BoraGrid::autoConnect();

    const bool displayGrid = MPlug( nodeObj, displayGridObj ).asBool();

	glPushAttrib( GL_ALL_ATTRIB_BITS );
    {
        if( displayGrid )
        {
            const bool dispGridX0 = MPlug( nodeObj, dispGridX0Obj ).asBool();
            const bool dispGridX1 = MPlug( nodeObj, dispGridX1Obj ).asBool();
            const bool dispGridY0 = MPlug( nodeObj, dispGridY0Obj ).asBool();
            const bool dispGridY1 = MPlug( nodeObj, dispGridY1Obj ).asBool();
            const bool dispGridZ0 = MPlug( nodeObj, dispGridZ0Obj ).asBool();
            const bool dispGridZ1 = MPlug( nodeObj, dispGridZ1Obj ).asBool();

            grid.glBeginNormalSpace();
            {
                grid.draw( dispGridX0, dispGridX1, dispGridY0, dispGridY1, dispGridZ0, dispGridZ1 );
            }
            grid.glEndNormalSpace();
        }
    }
	glPopAttrib();

    if( toUpdateAE )
    {
        MGlobal::executeCommand( MString( "updateAE " ) + nodeName );
        toUpdateAE = false;
    }
}

void BoraGrid::autoConnect()
{
    if( !isThe1stTimeDraw ) { return; }

    isThe1stTimeDraw = false;

    MDGModifier mod;

    // BoraGrid#.matrix -> BoraGridShape#.inXForm
    {
        MObject parentObj = dagNodeFn.parent( 0 );
        MFnDependencyNode parentXFormDGFn( parentObj );

        MPlug fromPlg = parentXFormDGFn.findPlug( "matrix" );
        MPlug toPlg   = MPlug( nodeObj, inXFormObj );

        if( !toPlg.isConnected() )
        {
            mod.connect( fromPlg, toPlg );
        }
    }

    // BoraGridShape#.output -> BoraGrid#.dynamics
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

MBoundingBox BoraGrid::boundingBox() const
{
    const AABB3f aabb = grid.boundingBox();

	MBoundingBox bBox;

    bBox.expand( AsMPoint( aabb.minPoint() ) );
    bBox.expand( AsMPoint( aabb.maxPoint() ) );

    return bBox;
}

bool BoraGrid::isBounded() const
{
	return true;
}

