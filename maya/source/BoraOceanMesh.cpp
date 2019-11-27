//-----------------//
// BoraOceanMesh.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2019.01.23                               //
//-------------------------------------------------------//

#include <BoraOceanMesh.h>
#include <BoraOcean.h>

MTypeId BoraOceanMesh::id( 0x300009 );
MString BoraOceanMesh::name( "BoraOceanMesh" );

MObject BoraOceanMesh::inOceanObj;
MObject BoraOceanMesh::inMeshObj;
MObject BoraOceanMesh::outMeshObj;

BoraOceanMesh::BoraOceanMesh()
{}

BoraOceanMesh::~BoraOceanMesh()
{}

void* BoraOceanMesh::creator()
{
    return new BoraOceanMesh();
}

void BoraOceanMesh::postConstructor()
{
    MPxNode::postConstructor();
}

MStatus BoraOceanMesh::initialize()
{
	MFnEnumAttribute    eAttr;
	MFnUnitAttribute    uAttr;
	MFnTypedAttribute   tAttr;
    MFnMatrixAttribute  mAttr;
	MFnNumericAttribute nAttr;

	MStatus status;

    inOceanObj = nAttr.create( "inOcean", "inOcean", MFnNumericData::kFloat, 0.f, &status );
    tAttr.setKeyable(1); 
	CHECK_MSTATUS( addAttribute( inOceanObj ) );

    inMeshObj = tAttr.create( "inMesh", "inMesh", MFnData::kMesh, &status );
    tAttr.setKeyable(1); tAttr.setReadable(false);
	CHECK_MSTATUS( addAttribute( inMeshObj ) );

    outMeshObj = tAttr.create( "outMesh", "outMesh", MFnData::kMesh, &status );
    tAttr.setKeyable(0);
	CHECK_MSTATUS( addAttribute( outMeshObj ) );    

    attributeAffects( inOceanObj, outMeshObj );
	attributeAffects( inMeshObj , outMeshObj );

    return MS::kSuccess;
}

MStatus BoraOceanMesh::compute( const MPlug& plug, MDataBlock& data )
{
    if( plug != outMeshObj ) return MS::kUnknownParameter;

    //std::cout<< "BoraOceanMesh::compute()" << std::endl;

    MObject connObj;
    GetConnectedNodeObject( thisMObject(), inOceanObj, true, connObj );

    MFnDependencyNode nodeFn(  connObj );
    BoraOcean* ocean = (BoraOcean*)nodeFn.userNode();

    const OceanTileVertexData& oceanData = ocean->getOceanTileVertexData();

    data.inputValue( inOceanObj );

    if( data.isClean( inMeshObj ) == false )
    {
        MObject meshObj = data.inputValue( inMeshObj ).asMeshTransformed();

        MFnMesh meshFn( meshObj );
        meshFn.getPoints( meshPoints );
    }    

    // output mesh
	{
        deformedPoints.setLength( meshPoints.length() );

        #pragma omp parallel for
        for( int n=0; n<deformedPoints.length(); ++n )
        {
            Vec3f wP( meshPoints[n].x, meshPoints[n].y, meshPoints[n].z );
            Vec3f G, P;
            float C;            

            oceanData.lerp( wP, &G, &P, NULL, NULL, &C );
            Vec3f disp = P - G;

            Vec3f def = wP + disp;
                        
            deformedPoints[n] = MPoint( def.x, def.y, def. z );
        }

        MObject meshObj = data.inputValue( inMeshObj ).asMeshTransformed();

		MFnMesh newMeshFn;
        MFnMeshData dataCreator;
        MObject newMeshData = dataCreator.create();
		newMeshFn.copy( meshObj, newMeshData );

		newMeshFn.setPoints( deformedPoints );
		//newMeshFn.setVertexColors( colors, colorIDs );

        data.outputValue( outMeshObj ).set( newMeshData );
	}

    return MS::kSuccess;
}

MStatus BoraOceanMesh::connectionMade( const MPlug& plug, const MPlug& otherPlug, bool asSrc )
{
	return MPxNode::connectionMade( plug, otherPlug, asSrc );
}

