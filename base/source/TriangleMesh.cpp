//------------------//
// TriangleMesh.cpp //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.03.09                               //
//-------------------------------------------------------//


#include <Bora.h>
//#include <Alembic/Abc/All.h>
//#include <Alembic/AbcGeom/All.h>
//#include <Alembic/AbcCoreHDF5/All.h>
//#include <Alembic/AbcCoreOgawa/All.h>
//#include <Alembic/AbcCoreFactory/All.h>

//using namespace Alembic;
//using namespace Alembic::Abc;
//using namespace Alembic::AbcGeom;
//using namespace Alembic::AbcCoreFactory;
//using namespace Alembic::AbcCoreAbstract;

BORA_NAMESPACE_BEGIN

TriangleMesh::TriangleMesh( const MemorySpace memorySpace )
{
    Particles::initialize( memorySpace );
    indices.initialize( 0, memorySpace );
}

TriangleMesh& TriangleMesh::operator=( const TriangleMesh& mesh )
{
    Particles::operator=( mesh );
    indices = mesh.indices;

    return (*this);
}

void TriangleMesh::reset()
{
    Particles::reset();
    indices.clear();
}

void TriangleMesh::clear()
{
    Particles::clear();
    indices.clear();
}

BORA_FUNC_QUAL
size_t TriangleMesh::numVertices() const
{
    return Particles::pos.size();
}

BORA_FUNC_QUAL
size_t TriangleMesh::numUVWs() const
{
    return Particles::uvw.size();
}

BORA_FUNC_QUAL
size_t TriangleMesh::numTriangles() const
{
    return (indices.size()/3);
}

BORA_FUNC_QUAL
AABB3f TriangleMesh::boundingBox() const
{
    return BoundingBox( Particles::pos );
}

void TriangleMesh::drawVertices() const
{
    const size_t numV = Particles::pos.size();
    if( numV == 0 ) { return; }

    glBegin( GL_POINTS );
    {
        for( size_t i=0; i<numV; ++i )
        {
            glVertex( Particles::pos[i] );
        }
    }
    glEnd();
}

void TriangleMesh::drawWireframe() const
{
    const size_t numV = Particles::pos.size();
    const size_t numT = indices.size() / 3;
    if( ( numV * numT ) == 0 ) { return; }

    glBegin( GL_LINES );
    {
        for( size_t i=0; i<numT; ++i )
        {
            const size_t& v0 = indices[3*i];
            const size_t& v1 = indices[3*i+1];
            const size_t& v2 = indices[3*i+2];

            const Vec3f& p0 = Particles::pos[v0];
            const Vec3f& p1 = Particles::pos[v1];
            const Vec3f& p2 = Particles::pos[v2];

            glVertex( p0 );   glVertex( p1 );
            glVertex( p1 );   glVertex( p2 );
            glVertex( p2 );   glVertex( p0 );
        }
    }
    glEnd();
}

void TriangleMesh::drawUVW() const
{
    const size_t numV = Particles::uvw.size();
    const size_t numT = numV / 3;
    if( ( numV * numT ) == 0 ) { return; }

    glBegin( GL_LINES );
    {
        for( size_t i=0; i<numT; ++i )
        {
            const size_t v0 = 3*i;
            const size_t v1 = v0+1;
            const size_t v2 = v1+1;

            const Vec3f& p0 = Particles::uvw[v0];
            const Vec3f& p1 = Particles::uvw[v1];
            const Vec3f& p2 = Particles::uvw[v2];

            glVertex( p0 );   glVertex( p1 );
            glVertex( p1 );   glVertex( p2 );
            glVertex( p2 );   glVertex( p0 );
        }
    }
    glEnd();
}

bool TriangleMesh::save( const char* filePathName ) const
{
    std::ofstream fout( filePathName, std::ios::out|std::ios::binary|std::ios::trunc );

    if( fout.fail() || !fout.is_open() )
    {
        COUT << "Error@TriangleMesh::save(): Failed to save file: " << filePathName << ENDL;
        return false;
    }

    Particles::write( fout );
    indices.write( fout );

    fout.close();

    return true;
}

bool TriangleMesh::load( const char* filePathName )
{
    TriangleMesh::reset();

    std::ifstream fin( filePathName, std::ios::in|std::ios::binary );

    if( fin.fail() )
    {
        COUT << "Error@TriangleMesh::load(): Failed to load file." << ENDL;
        return false;
    }

    Particles::read( fin );
    indices.read( fin );

    fin.close();

    return true;
}

//void TriangleMesh::exportAlembic( std::string& filePath )
//{
//    /*
//	if( filePath == "" ) { return; }
//	const Vec3fArray& vPoints = Particles::pos;
//	const IndexArray& vConnects = indices; 
//
//    int numV = vPoints.size();
//	int numC = vConnects.size();
//    int numT = numC / 3;
//    if( ( numV * numT ) == 0 ) { return; }
//
//	OArchive archive
//	(
//		AbcCoreOgawa::WriteArchive(),
//		filePath
//	);
//	uint32_t iTimeIndex = 0;
//
//	OXform xformObj( archive.getTop(), "xform", iTimeIndex );
//	OPolyMesh meshObj( xformObj, "mesh", iTimeIndex );
//	OPolyMeshSchema& mesh = meshObj.getSchema();
//
//	// TODO:normals? uv?
//	IntArray vCounts = IntArray(numT);
//	vCounts.setValueAll(3);
//
//	// Because need an int array, not an IndexArray
//	// and need to reverse vertex order
//	int vConnects1[numC];
//	for( int i=0; i<numT; ++i )
//	{
//		vConnects1[3*i  ] = vConnects[3*i+1];
//		vConnects1[3*i+1] = vConnects[3*i  ];
//		vConnects1[3*i+2] = vConnects[3*i+2];
//	}
//
//	// shape
//	OPolyMeshSchema::Sample mesh_sample
//	(
//		V3fArraySample   ( (const V3f*)vPoints.pointer(), numV ),
//		Int32ArraySample ( (const int*)vConnects1, numC ),
//		Int32ArraySample ( vCounts.pointer(), numT )
//	);
//
//	mesh.set( mesh_sample );
//    */
//}
//
//bool TriangleMesh::importAlembic( std::string& filePath )
//{   
//    /*
//	if( filePath == "" ) { return false; }
//	IFactory factory;
//	IFactory::CoreType coreType;
//	IArchive archive = factory.getArchive( filePath, coreType );
//
//	if( !archive.valid() )
//	{
//		COUT<<"Error@Bora::TriangleMesh::importAlembic(): Failed to open file."<<ENDL;
//		return false;
//	}
//
//	IObject topObj = archive.getTop();
//
//	IXform xformObj( topObj, topObj.getChildHeader(0).getName() );
//	IXformSchema xform = xformObj.getSchema();
//
//	// TODO: take into account xform transformations?
////	XformSample xs;
////	xform.get( xs, ISampleSelector((index_t)0) );
////
////	{
////		COUT<<"xs[0].isTranslateOp(): "<<xs[0].isTranslateOp()<<ENDL;;
////		COUT<<"xs[1].isRotateZOp(): "<<xs[1].isRotateZOp()<<ENDL;;
////		COUT<<"xs[2].isRotateYOp(): "<<xs[2].isRotateYOp()<<ENDL;;
////		COUT<<"xs[3].isRotateXOp(): "<<xs[3].isRotateXOp()<<ENDL;;
////		COUT<<"xs[4].isScaleOp(): "<<xs[4].isScaleOp()<<ENDL;;
////
////		xs[0].getTranslate();
//////		xs[1].getZRotation();
//////		xs[2].getYRotation();
//////		xs[3].getXRotation();
////		xs[4].getScale();
////
////	}
//
//	IPolyMesh meshObj( xformObj, xformObj.getChildHeader(0).getName() );
//	const IPolyMeshSchema& mesh = meshObj.getSchema();
//
//	IPolyMeshSchema::Sample ms;
//	mesh.get( ms, ISampleSelector((index_t)0) );
//
//	const P3fArraySample& vPoints = *( ms.getPositions() );
//	int numV = vPoints.size();
//	Particles::pos.resize(numV);
//	for( int i=0; i<numV; ++i )
//	{
//		Vec3f& to = Particles::pos[i];
//		const V3f& from = vPoints[i];
//		to.x = from.x;
//		to.y = from.y;
//		to.z = from.z;
//	}
//
//	const Int32ArraySample& vConnects = *( ms.getFaceIndices() );
//	int numC = vConnects.size();
//	indices.resize(numC);
//	for( int i=0; i<(numC/3); ++i )
//	{
//		indices[3*i  ] = vConnects[3*i+1];
//		indices[3*i+1] = vConnects[3*i  ];
//		indices[3*i+2] = vConnects[3*i+2];
//	}
//    */
//	return true;
//}

std::ostream& operator<<( std::ostream& os, const TriangleMesh& object )
{
    os << "<TriangleMesh>" << ENDL;
    os << " # of vertices: " << object.numVertices() << ENDL;
    os << " # of triangles: " << object.numTriangles() << ENDL;
    os << " # of UVWs: " << object.numUVWs() << ENDL;
    os << ENDL;
    return os;
}

BORA_NAMESPACE_END

