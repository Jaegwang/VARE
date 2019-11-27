//-------------------//
// BoraHeightMerge.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2019.03.28                               //
//-------------------------------------------------------//

#include <BoraHeightMerge.h>

MTypeId BoraHeightMerge::id( 0x300010 );
MString BoraHeightMerge::name( "BoraHeightMerge" );
MString	BoraHeightMerge::drawDbClassification( "drawdb/geometry/BoraHeightMerge" );
MString	BoraHeightMerge::drawRegistrantId( "BoraHeightMergeNodePlugin" );

MObject BoraHeightMerge::inHeightMeshsObj;
MObject BoraHeightMerge::inDomainMeshObj;
MObject BoraHeightMerge::outMeshObj;

BoraHeightMerge::BoraHeightMerge()
{}

BoraHeightMerge::~BoraHeightMerge()
{}

void* BoraHeightMerge::creator()
{
    return new BoraHeightMerge();
}

void BoraHeightMerge::postConstructor()
{
    MPxNode::postConstructor();
}

MStatus BoraHeightMerge::initialize()
{
	MFnEnumAttribute    eAttr;
	MFnUnitAttribute    uAttr;
	MFnTypedAttribute   tAttr;
    MFnMatrixAttribute  mAttr;
	MFnNumericAttribute nAttr;

	MStatus status;

    inHeightMeshsObj = tAttr.create( "inMeshs", "inMeshs", MFnData::kMesh, &status );
    tAttr.setArray(1); tAttr.setKeyable(1); tAttr.setReadable(false); tAttr.setStorable(false);
    tAttr.setDisconnectBehavior( MFnAttribute::kDelete );
	CHECK_MSTATUS( addAttribute( inHeightMeshsObj ) );

    inDomainMeshObj = tAttr.create( "inDomainMesh", "inDomainMesh", MFnData::kMesh, &status );
    tAttr.setKeyable(1); tAttr.setReadable(false); tAttr.setStorable(false);    
    tAttr.setDisconnectBehavior( MFnAttribute::kDelete );
	CHECK_MSTATUS( addAttribute( inDomainMeshObj ) );

    outMeshObj = tAttr.create( "outMesh", "outMesh", MFnData::kMesh, &status );
    tAttr.setKeyable(0);
	CHECK_MSTATUS( addAttribute( outMeshObj ) );


	attributeAffects( inHeightMeshsObj , outMeshObj );
    attributeAffects( inDomainMeshObj  , outMeshObj );


    return MS::kSuccess;
}

MStatus BoraHeightMerge::compute( const MPlug& plug, MDataBlock& data )
{
    if( plug != outMeshObj ) return MS::kUnknownParameter;

    vertices.clear();
    triangles.clear();    

    MArrayDataHandle inMeshsHnd = data.inputArrayValue( inHeightMeshsObj );

    float total_area(0.f);
    int count(0); 

    for( int n=0; n<inMeshsHnd.elementCount(); ++n, inMeshsHnd.next() )
    {
        MObject meshObject = inMeshsHnd.inputValue().asMeshTransformed();
        MFnMesh meshFn( meshObject );

        MPointArray points_world;
        meshFn.getPoints( points_world, MSpace::kWorld );

        for( int x=0; x<points_world.length(); ++x )
        {
            MPoint p = points_world[x];
            vertices.append( Vec3f( p.x, p.y, p.z ) );

            if( min.x > p.x ) min.x = p.x;
            if( min.y > p.z ) min.y = p.z;

            if( max.x < p.x ) max.x = p.x;
            if( max.y < p.z ) max.y = p.z;            
        }

        MIntArray triCount, triVert;
        meshFn.getTriangles( triCount, triVert );

        for( int x=0; x<triVert.length()/3; ++x )
        {
            triangles.append( triVert[x*3+0]+count );
            triangles.append( triVert[x*3+1]+count );
            triangles.append( triVert[x*3+2]+count );

            const MPoint& p0 = points_world[triVert[x*3+0]];
            const MPoint& p1 = points_world[triVert[x*3+1]];
            const MPoint& p2 = points_world[triVert[x*3+2]];

            const float area = ((p1-p0)^(p2-p0)).length();
            total_area += area;
        }

        count += points_world.length();
    }

    total_area /= (float)(triangles.size()/3);
    voxelSize = Pow( total_area, 0.5f ) * 2.f;


    hashGrid.initialize( min, max, voxelSize );
    hashGrid.clear();

    for( int x=0; x<triangles.size()/3; ++x )
    {
        const Vec3f& p0 = vertices[ triangles[x*3+0] ];
        const Vec3f& p1 = vertices[ triangles[x*3+1] ];
        const Vec3f& p2 = vertices[ triangles[x*3+2] ];

        int i0, j0, i1, j1, i2, j2;
        hashGrid.indices( i0, j0, p0.x, p0.z );
        hashGrid.indices( i1, j1, p1.x, p1.z );
        hashGrid.indices( i2, j2, p2.x, p2.z );

        int min_i = Min( Min(i0,i1),i2 );
        int min_j = Min( Min(j0,j1),j2 );
        int max_i = Max( Max(i0,i1),i2 );
        int max_j = Max( Max(j0,j1),j2 );

        for( int j=min_j; j<=max_j; ++j )
        for( int i=min_i; i<=max_i; ++i )
        {
            size_t hh = hashGrid.hash( i, j );
            hashGrid[hh].push_back( x );
        }
    }


    {
        MObject domainMeshObj = data.inputValue( inDomainMeshObj ).asMeshTransformed();
        MFnMesh domainMesh( domainMeshObj );

        MPointArray points_domain;
        domainMesh.getPoints( points_domain, MSpace::kWorld );

        auto kernel = [&]( size_t n )
        {
            MPoint& p = points_domain[n];
            Vec3f pos( p.x, p.y, p.z );

            float height = pos.y;            

            size_t hh = hashGrid.hash( (float)p.x, (float)p.z );

            for( int t=0; t<hashGrid[hh].size(); ++t )
            {
                int q = hashGrid[hh][t];
                const Vec3f& p0 = vertices[ triangles[q*3+0] ];
                const Vec3f& p1 = vertices[ triangles[q*3+1] ];
                const Vec3f& p2 = vertices[ triangles[q*3+2] ];

                Vec3f inter;
                if( rayTriangleIntersection( inter, pos, Vec3f(0.f,1.f,0.f), p0, p1, p2 ) )
                {
                    height = Max( height, inter.y );                    
                }
            }

            p.y = height;
        };

        LaunchCudaHost( kernel, 0, points_domain.length() );

		MFnMesh newMeshFn;
        MFnMeshData dataCreator;        
        MObject newMeshData = dataCreator.create();
		newMeshFn.copy( domainMeshObj, newMeshData );

		newMeshFn.setPoints( points_domain );

        data.outputValue( outMeshObj ).set( newMeshData );
    }

    return MS::kSuccess;
}

void BoraHeightMerge::draw( M3dView& view, const MDagPath& path, M3dView::DisplayStyle style, M3dView::DisplayStatus status )
{
    view.beginGL();

    draw();

    view.endGL();
}

void BoraHeightMerge::draw( int drawingMode )
{
    /*
    glPushAttrib( GL_ALL_ATTRIB_BITS );

    glBegin( GL_POINTS );

    glVertex3f( min.x, 2.f, min.y );
    glVertex3f( max.x, 2.f, max.y );

    glEnd();

    glBegin( GL_TRIANGLES );

    for( int x=0; x<triangles.size()/3; ++x )
    {
        Vec3f& p0 = vertices[triangles[x*3+0]];
        Vec3f& p1 = vertices[triangles[x*3+1]];
        Vec3f& p2 = vertices[triangles[x*3+2]];

        glVertex3f( p0.x, p0.y, p0.z );
        glVertex3f( p1.x, p1.y, p1.z );
        glVertex3f( p2.x, p2.y, p2.z );

    }

    glEnd();

    glPopAttrib();
    */
}

MBoundingBox BoraHeightMerge::boundingBox() const
{
    const AABB3f aabb( Vec3f(-100.f), Vec3f(100.f) );

	MBoundingBox bBox;

    bBox.expand( AsMPoint( aabb.minPoint() ) );
    bBox.expand( AsMPoint( aabb.maxPoint() ) );

    return bBox;
}

MStatus BoraHeightMerge::connectionMade( const MPlug& plug, const MPlug& otherPlug, bool asSrc )
{
	return MPxNode::connectionMade( plug, otherPlug, asSrc );
}

bool
BoraHeightMerge::rayTriangleIntersection( Vec3f& P, const Vec3f& P0, const Vec3f& V, const Vec3f& T0, const Vec3f& T1, const Vec3f& T2 )
{  
    const Vec3f N = ( (T1-T0)^(T2-T0) ).normalized();
    const float d = -(T0*N);

    const Vec3f NV = V.normalized();

    const float t = -(P0*N + d)/(NV*N);

    P = P0 + NV*t;
    if( (P-P0)*NV < 0.f ) return false;


    Vec3f d0 = (T1-T0).normalized();
    d0 = T0 + d0*(d0*(P-T0));

    if( (P-d0) * (T2-d0) < 0.f ) return false;

    Vec3f d1 = (T2-T1).normalized();
    d1 = T1 + d1*(d1*(P-T1));

    if( (P-d1) * (T0-d1) < 0.f ) return false;

    Vec3f d2 = (T0-T2).normalized();
    d2 = T2 + d2*(d2*(P-T2));

    if( (P-d2) * (T1-d2) < 0.f ) return false;

    return true;
}

