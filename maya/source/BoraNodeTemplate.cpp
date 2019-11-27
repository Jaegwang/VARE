//----------------------//
// BoraNodeTemplate.cpp //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.01.24                               //
//-------------------------------------------------------//

#include <BoraNodeTemplate.h>

MTypeId BoraNodeTemplate::id( 0x300000 );
MString BoraNodeTemplate::name( "BoraNodeTemplate" );
MString	BoraNodeTemplate::drawDbClassification( "drawdb/geometry/BoraNodeTemplate" );
MString	BoraNodeTemplate::drawRegistrantId( "BoraNodeTemplateNodePlugin" );

MObject BoraNodeTemplate::inTimeObj;
MObject BoraNodeTemplate::inXFormObj;
MObject BoraNodeTemplate::outputObj;

void* BoraNodeTemplate::creator()
{
	return new BoraNodeTemplate();
}

void BoraNodeTemplate::postConstructor()
{
	MPxNode::postConstructor();

	nodeObj = thisMObject();
	nodeFn.setObject( nodeObj );
	dagNodeFn.setObject( nodeObj );
	nodeFn.setName( "BoraNodeTemplateShape#" );
}

MStatus BoraNodeTemplate::initialize()
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

    inXFormObj = mAttr.create( "inXForm", "inXForm", MFnMatrixAttribute::kDouble, &s );
	mAttr.setHidden(1);
	CHECK_MSTATUS( addAttribute( inXFormObj ) );

    outputObj = nAttr.create( "output", "output", MFnNumericData::kFloat, 0.f, &s );
	nAttr.setHidden(1);
    CHECK_MSTATUS( addAttribute( outputObj ) );

    attributeAffects( inTimeObj,  outputObj );
    attributeAffects( inXFormObj, outputObj );

	return MS::kSuccess;
}

MStatus BoraNodeTemplate::compute( const MPlug& plug, MDataBlock& block )
{
	if( plug != outputObj ) { return MS::kUnknownParameter; }

	blockPtr = &block;
	nodeName = nodeFn.name();
	MThreadUtils::syncNumOpenMPThreads();

	float currentTime = (float)block.inputValue( inTimeObj ).asTime().as( MTime::uiUnit() );

    MMatrix m = block.inputValue( inXFormObj ).asMatrix();
    cout << m << endl;

	block.outputValue( outputObj ).set( 0.f );
	block.setClean( plug );

    toUpdateAE = true;

	return MS::kSuccess;
}

void BoraNodeTemplate::draw( M3dView& view, const MDagPath& path, M3dView::DisplayStyle style, M3dView::DisplayStatus status )
{
	view.beginGL();
	{
		draw();
	}
	view.endGL();
}

void BoraNodeTemplate::draw( int drawingMode )
{
    BoraNodeTemplate::autoConnect();

	glPushAttrib( GL_ALL_ATTRIB_BITS );
    {
        glLineWidth( 10 );
        glColor3f( 1, 0, 0 );

        glBegin( GL_LINES );
        {
            glVertex( Vec3f( 0, 0, 0 ) );
            glVertex( Vec3f( 1, 1, 1 ) );
        }
        glEnd();
    }
	glPopAttrib();

    if( toUpdateAE )
    {
        MGlobal::executeCommand( MString( "updateAE " ) + nodeName );
        toUpdateAE = false;
    }
}

void BoraNodeTemplate::autoConnect()
{
    if( !isThe1stTimeDraw ) { return; }

    isThe1stTimeDraw = false;

    MDGModifier mod;

    // time1.outTime -> BoraNodeTemplateShape#.inTime
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

    // BoraNodeTemplate#.matrix -> BoraNodeTemplateShape#.inXForm
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

    // BoraNodeTemplateShape#.output -> BoraNodeTemplate#.dynamics
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

MBoundingBox BoraNodeTemplate::boundingBox() const
{
	MBoundingBox bBox;

    bBox.expand( AsMPoint( Vec3f(0) ) );
    bBox.expand( AsMPoint( Vec3f(1) ) );

    return bBox;
}

bool BoraNodeTemplate::isBounded() const
{
	return true;
}

