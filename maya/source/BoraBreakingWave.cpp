//--------------------//
// BoraBreakingWave.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2017.12.19                               //
//-------------------------------------------------------//

#include <BoraBreakingWave.h>

MTypeId BoraBreakingWave::id( 0x300008 );
MString BoraBreakingWave::name( "BoraBreakingWave" );

MObject BoraBreakingWave::inMeshObj;
MObject BoraBreakingWave::inProfileLinesObj;
MObject BoraBreakingWave::outMeshObj;
MObject BoraBreakingWave::widthObj;

BoraBreakingWave::BoraBreakingWave()
{}

BoraBreakingWave::~BoraBreakingWave()
{}

void* BoraBreakingWave::creator()
{
    return new BoraBreakingWave();
}

void BoraBreakingWave::postConstructor()
{
    MPxNode::postConstructor();
}

MStatus BoraBreakingWave::initialize()
{
	MFnEnumAttribute    eAttr;
	MFnUnitAttribute    uAttr;
	MFnTypedAttribute   tAttr;
    MFnMatrixAttribute  mAttr;
	MFnNumericAttribute nAttr;

	MStatus status;

    inMeshObj = tAttr.create( "inMesh", "inMesh", MFnData::kMesh, &status );
    tAttr.setKeyable(1); tAttr.setReadable(false);
	CHECK_MSTATUS( addAttribute( inMeshObj ) );

    inProfileLinesObj = tAttr.create( "inProfiles", "inProfiles", MFnData::kNurbsCurve, &status );
    tAttr.setArray(1); tAttr.setKeyable(1); tAttr.setReadable(false); tAttr.setStorable(false);
    tAttr.setDisconnectBehavior( MFnAttribute::kDelete );
	CHECK_MSTATUS( addAttribute( inProfileLinesObj ) );

    widthObj = nAttr.create( "width", "width", MFnNumericData::kFloat, 3.f, &status );
    nAttr.setMin(0.f);
	CHECK_MSTATUS( addAttribute( widthObj ) );

    outMeshObj = tAttr.create( "outMesh", "outMesh", MFnData::kMesh, &status );
    tAttr.setKeyable(0);
	CHECK_MSTATUS( addAttribute( outMeshObj ) );


	attributeAffects( inMeshObj , outMeshObj );
	attributeAffects( inProfileLinesObj , outMeshObj );    
    attributeAffects( widthObj, outMeshObj );

    return MS::kSuccess;
}

MStatus BoraBreakingWave::compute( const MPlug& plug, MDataBlock& data )
{
    if( plug != outMeshObj ) return MS::kUnknownParameter;

    std::cout<< "BoraBreakingWave::compute()" << std::endl;

    MArrayDataHandle inProfilesHnd = data.inputArrayValue( inProfileLinesObj );
	int profileCount(0);
    
    for( int n=0; n<inProfilesHnd.elementCount(); ++n, inProfilesHnd.next() )
    {
        profileCount++;
    }

    inProfilesHnd.jumpToElement(0);
    breakingWave.resize( profileCount );

    for( int n=0; n<inProfilesHnd.elementCount(); ++n, inProfilesHnd.next() )
    {
        MObject wpObject = inProfilesHnd.inputValue().asNurbsCurveTransformed();
        MFnNurbsCurve curveFn(wpObject);

        const double len = curveFn.length();
        //const int N = curveFn.numCVs()*4;
        const int N = len/0.5f;

        SplineCurve* pCurve = breakingWave.getControlCurve( n );

        pCurve->cp.initialize( N, kUnified );                

        for( int i=0; i<N; ++i )
        {
            double d = (double(i)/double(N-1)) * len;
            double u = curveFn.findParamFromLength( d );

            MPoint R;
            curveFn.getPointAtParam( u, R );

            pCurve->cp[i] = Vec3f( R.x, R.y, R.z );
        }
    }

    if( data.isClean( inMeshObj ) == false )
    {
        MObject meshObj = data.inputValue( inMeshObj ).asMeshTransformed();

		MPointArray meshPoints;

        MFnMesh meshFn( meshObj );
        meshFn.getPoints( meshPoints );

        inputPoints.initialize( meshPoints.length(), kUnified );

        for( int n=0; n<meshPoints.length(); ++n )
        {
            const MPoint& p = meshPoints[n];
            inputPoints[n] = Vec3f( p.x, p.y, p.z );
        }
    }

	const float rad = data.inputValue( widthObj ).asFloat();
    breakingWave.wave( deformedPoints, inputPoints, rad );

    // output mesh
	{
        MObject meshObj = data.inputValue( inMeshObj ).asMeshTransformed();

		MPointArray meshPoints( deformedPoints.size() );
        for( int n=0; n<deformedPoints.size(); ++n )
        {
            const Vec3f& p = deformedPoints[n];
            meshPoints[n] = MPoint( p.x, p.y, p.z );
        }

		MFnMesh newMeshFn;
        MFnMeshData dataCreator;        
        MObject newMeshData = dataCreator.create();
		newMeshFn.copy( meshObj, newMeshData );

		newMeshFn.setPoints( meshPoints );
		//newMeshFn.setVertexColors( colors, colorIDs );

        data.outputValue( outMeshObj ).set( newMeshData );
	}

    return MS::kSuccess;
}

MStatus BoraBreakingWave::connectionMade( const MPlug& plug, const MPlug& otherPlug, bool asSrc )
{
	return MPxNode::connectionMade( plug, otherPlug, asSrc );
}

