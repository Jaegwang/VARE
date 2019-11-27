//----------------------//
// Bora_ConvertToLevelSet_FMM.cpp //
//-------------------------------------------------------//
// author: Julie Jang @ Dexter Studios                   //
// last update: 2018.04.10                               //
//-------------------------------------------------------//

#include <Bora_ConvertToLevelSet_FMM.h>

MTypeId Bora_ConvertToLevelSet_FMM::id( 0x300005 );
MString Bora_ConvertToLevelSet_FMM::name( "Bora_ConvertToLevelSet_FMM" );

MString	Bora_ConvertToLevelSet_FMM::drawDbClassification( "drawdb/geometry/Bora_ConvertToLevelSet_FMM" );
MString	Bora_ConvertToLevelSet_FMM::drawRegistrantId( "ZarVisForMayaPlugin" );

MObject Bora_ConvertToLevelSet_FMM::inputObj;
MObject Bora_ConvertToLevelSet_FMM::inXFormObj;
MObject Bora_ConvertToLevelSet_FMM::outputObj;
MObject Bora_ConvertToLevelSet_FMM::subdivisionObj;
MObject Bora_ConvertToLevelSet_FMM::inMeshObj;

Bora_ConvertToLevelSet_FMM::Bora_ConvertToLevelSet_FMM()
{
	toUpdateAE = true;
}

Bora_ConvertToLevelSet_FMM::~Bora_ConvertToLevelSet_FMM()
{
    // nothing to do
}

void*
Bora_ConvertToLevelSet_FMM::creator()
{
    return new Bora_ConvertToLevelSet_FMM();
}

void
Bora_ConvertToLevelSet_FMM::postConstructor()
{
    nodeObj = thisMObject();
    nodeFn.setObject( nodeObj );
    dagNodeFn.setObject( nodeObj );
    nodeFn.setName( "Bora_ConvertToLevelSet_FMMShape#" );
}

MStatus
Bora_ConvertToLevelSet_FMM::initialize()
{
	MStatus s = MS::kSuccess;

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
Bora_ConvertToLevelSet_FMM::compute( const MPlug& plug, MDataBlock& block )
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
	AABB3f aabb = AABB3f( Vec3f(-5.f), Vec3f(5.f) );
	Grid grid( subdivision, aabb );
	_lvs.setGrid( grid );
	_stt.setGrid( grid );
	//_lvsSolver.reset();
	_lvsSolver.set( grid );
	_lvsSolver.addMesh( _lvs, _stt, _triMesh, BORA_LARGE, BORA_LARGE );
	_lvsSolver.finalize();

	MMatrix m = block.inputValue( inXFormObj ).asMatrix();

    childChanged( MPxSurfaceShape::kBoundingBoxChanged );

    MDataHandle outputHnd = block.outputValue( outputObj );
    outputHnd.set( 0.f );

    block.setClean( plug );

    toUpdateAE = true;

    return MS::kSuccess;
}

MBoundingBox Bora_ConvertToLevelSet_FMM::boundingBox() const
{
    //const AABB3f aabb = _densityField.boundingBox();

    MBoundingBox bBox;

//    bBox.expand( AsMPoint( aabb.minPoint() ) );
//    bBox.expand( AsMPoint( aabb.maxPoint() ) );
    bBox.expand( MPoint( -10000, -10000, -10000 ) );
    bBox.expand( MPoint(  10000,  10000,  10000 ) );

    return bBox;
}

bool Bora_ConvertToLevelSet_FMM::isBounded() const
{
    return true;
}

void Bora_ConvertToLevelSet_FMM::draw( int drawingMode )
{
    _autoConnect();

    const float end = MPlug( nodeObj, outputObj ).asFloat();

//	float xSlice = MPlug( thisObj, xSliceObj ).asFloat();
//	float ySlice = MPlug( thisObj, ySliceObj ).asFloat();
//	float zSlice = MPlug( thisObj, zSliceObj ).asFloat();

	Vec3i whichSlice( 1, 1, 1 );
//	ZFloat3 sliceRatio( xSlice, ySlice, zSlice );

	glPushAttrib( GL_ALL_ATTRIB_BITS );

	drawSlice( whichSlice, Vec3f(0, 0, 0) );

	glPopAttrib();

//    const float end = MPlug( nodeObj, outputObj ).asFloat();
//
//    if( toUpdateAE )
//    {
//        MGlobal::executeCommand( MString( "updateAE " ) + nodeName );
//        toUpdateAE = false;
//    }
}

void
Bora_ConvertToLevelSet_FMM::drawSlice( const Vec3i& whichSlice, const Vec3f& sliceRatio,
                           bool smoothPosArea, const Vec4f& farPos, const Vec4f& nearPos,
                           bool smoothNegArea, const Vec4f& farNeg, const Vec4f& nearNeg,
                           float elementSize ) const
{
	int _iMax = _lvs.nx()-1;
	int _jMax = _lvs.ny()-1;
	int _kMax = _lvs.nz()-1;

	if( whichSlice[0] )
	{
		_drawXSlice( Clamp( (int)(_iMax*sliceRatio[0]), 0, _iMax ),
                     smoothPosArea, farPos, nearPos,
                     smoothNegArea, farNeg, nearNeg,
                     elementSize );
	}

	if( whichSlice[1] )
	{
		_drawYSlice( Clamp( (int)(_jMax*sliceRatio[1]), 0, _jMax ),
                     smoothPosArea, farPos, nearPos,
                     smoothNegArea, farNeg, nearNeg,
                     elementSize );
	}

	if( whichSlice[2] )
	{
		_drawZSlice( Clamp( (int)(_kMax*sliceRatio[2]), 0, _kMax ),
                     smoothPosArea, farPos, nearPos,
                     smoothNegArea, farNeg, nearNeg,
                     elementSize );
	}
}


void
Bora_ConvertToLevelSet_FMM::_drawXSlice( int i,
                             bool smoothPosArea, const Vec4f& farPos, const Vec4f& nearPos,
                             bool smoothNegArea, const Vec4f& farNeg, const Vec4f& nearNeg,
                             float elementSize ) const
{
	const int numElems = _lvs.numCells();
	if( numElems<1 ) { return; }

	const float h = Clamp( elementSize, 0.f, 1.f );
	int _iMax = _lvs.nx()-1;
	int _jMax = _lvs.ny()-1;
	int _kMax = _lvs.nz()-1;
	AABB3f aabb = _lvs.boundingBox();
    const float Lx = aabb.width(0);
    const float Ly = aabb.width(1);
    const float Lz = aabb.width(2);
	float _dx = Lx/float(_lvs.nx());
	float _dy = Ly/float(_lvs.ny());
	float _dz = Lz/float(_lvs.nz());
	const float hDx = 0.5f*h*_dx;
	const float hDy = 0.5f*h*_dy;
	const float hDz = 0.5f*h*_dz;

	Vec3f p;

	glBegin( GL_QUADS );

		for( int k=0; k<_kMax+1; k++ )
		for( int j=0; j<_jMax+1; j++ )
		{{
			const float& v = _lvs(i,j,k);

			const float  a = v / ( _lvs.minValue() + EPSILON );
			const float _a = 1 - a;
			const float  b = v / ( _lvs.maxValue() + EPSILON );
			const float _b = 1 - b;

			if( v<0 ) {
				if( smoothNegArea ) { 
					glColor( a*farNeg + _a*nearNeg ); 
				}
				else                { glColor( farNeg ); }
			} else {
				if( smoothPosArea ) { glColor( b*farPos + _b*nearPos ); }
				else                { glColor( farPos ); }
			}

			p = _lvs.worldPoint( Vec3f(i,j,k) );
		
			glVertex( p.x, p.y-hDy, p.z-hDz );
			glVertex( p.x, p.y+hDy, p.z-hDz );
			glVertex( p.x, p.y+hDy, p.z+hDz );
			glVertex( p.x, p.y-hDy, p.z+hDz );

		}}

	glEnd();
}

void
Bora_ConvertToLevelSet_FMM::_drawYSlice( int j,
                             bool smoothPosArea, const Vec4f& farPos, const Vec4f& nearPos,
                             bool smoothNegArea, const Vec4f& farNeg, const Vec4f& nearNeg,
                             float elementSize ) const
{
	const int numElems = _lvs.numCells();
	if( numElems<1 ) { return; }

	const float h = Clamp( elementSize, 0.f, 1.f );
	int _iMax = _lvs.nx()-1;
	int _jMax = _lvs.ny()-1;
	int _kMax = _lvs.nz()-1;
	AABB3f aabb = _lvs.boundingBox();
    const float Lx = aabb.width(0);
    const float Ly = aabb.width(1);
    const float Lz = aabb.width(2);
	float _dx = Lx/float(_lvs.nx());
	float _dy = Ly/float(_lvs.ny());
	float _dz = Lz/float(_lvs.nz());
	const float hDx = 0.5f*h*_dx;
	const float hDy = 0.5f*h*_dy;
	const float hDz = 0.5f*h*_dz;

	Vec3f p;

	glBegin( GL_QUADS );

		for( int k=0; k<_kMax+1; k++ )
		for( int i=0; i<_iMax+1; i++ )
		{{
			const float& v = _lvs(i,j,k);

			const float  a = v / ( _lvs.minValue() + EPSILON );
			const float _a = 1 - a;
			const float  b = v / ( _lvs.maxValue() + EPSILON );
			const float _b = 1 - b;

			if( v<0 ) {
				if( smoothNegArea ) { glColor( a*farNeg + _a*nearNeg ); }
				else                { glColor( farNeg ); }
			} else {
				if( smoothPosArea ) { glColor( b*farPos + _b*nearPos ); }
				else                { glColor( farPos ); }
			}

			p = _lvs.worldPoint( Vec3f(i,j,k) );	

			glVertex( p.x-hDx, p.y, p.z-hDz );
			glVertex( p.x+hDx, p.y, p.z-hDz );
			glVertex( p.x+hDx, p.y, p.z+hDz );
			glVertex( p.x-hDx, p.y, p.z+hDz );
		}}

	glEnd();
}

void
Bora_ConvertToLevelSet_FMM::_drawZSlice( int k,
                             bool smoothPosArea, const Vec4f& farPos, const Vec4f& nearPos,
                             bool smoothNegArea, const Vec4f& farNeg, const Vec4f& nearNeg,
                             float elementSize ) const
{
	const int numElems = _lvs.numCells();
	if( numElems<1 ) { return; }

	const float h = Clamp( elementSize, 0.f, 1.f );
	int _iMax = _lvs.nx()-1;
	int _jMax = _lvs.ny()-1;
	int _kMax = _lvs.nz()-1;
	AABB3f aabb = _lvs.boundingBox();
    const float Lx = aabb.width(0);
    const float Ly = aabb.width(1);
    const float Lz = aabb.width(2);
	float _dx = Lx/float(_lvs.nx());
	float _dy = Ly/float(_lvs.ny());
	float _dz = Lz/float(_lvs.nz());
	const float hDx = 0.5f*h*_dx;
	const float hDy = 0.5f*h*_dy;
	const float hDz = 0.5f*h*_dz;

	Vec3f p;

	glBegin( GL_QUADS );

		for( int j=0; j<_jMax+1; j++ )
		for( int i=0; i<_iMax+1; i++ )
		{{
			const float& v = _lvs(i,j,k);

			const float  a = v / ( _lvs.minValue() + EPSILON );
			const float _a = 1 - a;
			const float  b = v / ( _lvs.maxValue() + EPSILON );
			const float _b = 1 - b;

			if( v<0 ) {
				if( smoothNegArea ) { glColor( a*farNeg + _a*nearNeg ); }
				else                { glColor( farNeg ); }
			} else {
				if( smoothPosArea ) { glColor( b*farPos + _b*nearPos ); }
				else                { glColor( farPos ); }
			}

			p = _lvs.worldPoint( Vec3f(i,j,k) );	

			glVertex( p.x-hDx, p.y-hDy, p.z );
			glVertex( p.x+hDx, p.y-hDy, p.z );
			glVertex( p.x+hDx, p.y+hDy, p.z );
			glVertex( p.x-hDx, p.y+hDy, p.z );
		}}

	glEnd();
}

void Bora_ConvertToLevelSet_FMM::_autoConnect()
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

