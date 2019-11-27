//--------------//
// Surfacer.cpp //
//-------------------------------------------------------//
// author: Julie Jang @ Dexter Digital                   //
// last update: 2018.03.28                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

Surfacer::Surfacer()
{
	reset();
}

Surfacer::Surfacer( const Surfacer& source )
{
	*this = source;
}

void
Surfacer::reset()
{
	_edg.clear();
	_lvs.clear();
}

Surfacer&
Surfacer::operator=( const Surfacer& source )
{
	_edg = source._edg;
	_lvs = source._lvs;
	return (*this);
}

bool
Surfacer::compute( const ScalarDenseField& lvs, TriangleMesh& mesh )
{
	// phi = ZScalarField, mesh = ZMesh, 0 = 0

	const int iMax = lvs.nx()-1;
	const int jMax = lvs.ny()-1;
	const int kMax = lvs.nz()-1;

	if( iMax<=0 && jMax<=0 && kMax<=0 ) { return true; }

	_edg.initialize( lvs.numVoxels() );

	_edg.setValueAll( Vec3i(-1) );

	int			Node[8];
	float		Phi[8];
	Vec3f		basePos;
	Vec3f		pt;
	Vec3fArray 	positions;
	IndexArray	polyConnections;

	int vertCount = 0;
	int threeEdges[3];
	int threeVerts[3];

	AABB3f aabb = lvs.boundingBox();
	float Lx = aabb.width(0);
	float Ly = aabb.width(1);
	float Lz = aabb.width(2);
	float rx = lvs.nx()/Lx;
	float ry = lvs.ny()/Ly;
	float rz = lvs.nz()/Lz;


	for( int k=0; k<kMax; ++k )
	for( int j=0; j<jMax; ++j )
	for( int i=0; i<iMax; ++i )
	{{{
		basePos = lvs.worldPoint(Vec3f(i,j,k));

		// eight corner indices for this cell
		int idx = lvs.cellIndex(i,j,k);
		Node[0] = idx;
		Node[1] = lvs.i1(idx);
		Node[2] = lvs.k1(Node[1]);
		Node[3] = lvs.k1(Node[0]);
		Node[4] = lvs.j1(Node[0]);
		Node[5] = lvs.j1(Node[1]);
		Node[6] = lvs.k1(Node[5]);
		Node[7] = lvs.k1(Node[4]);

		// eight corner level-set values for this cell
		for( int l=0; l<8; l++ ) { Phi[l] = lvs[ Node[l] ]; }

		// twelve candiate edgs which new vertices are created on
		int *vertOnEdge[12];
		vertOnEdge[ 0] = &_edg[ Node[0] ][0];
		vertOnEdge[ 1] = &_edg[ Node[1] ][2];
		vertOnEdge[ 2] = &_edg[ Node[3] ][0];
		vertOnEdge[ 3] = &_edg[ Node[0] ][2];
		vertOnEdge[ 4] = &_edg[ Node[4] ][0];
		vertOnEdge[ 5] = &_edg[ Node[5] ][2];
		vertOnEdge[ 6] = &_edg[ Node[7] ][0];
		vertOnEdge[ 7] = &_edg[ Node[4] ][2];
		vertOnEdge[ 8] = &_edg[ Node[0] ][1];
		vertOnEdge[ 9] = &_edg[ Node[1] ][1];
		vertOnEdge[10] = &_edg[ Node[2] ][1];
		vertOnEdge[11] = &_edg[ Node[3] ][1];

		// Which vertices are inside?
		// If i-th vertex is inside, mark '1' at i-th bit. of 'iFlagIndex'.
		int iFlagIndex = 0;
		for( int l=0; l<8; l++ ) { if( Phi[l] <= 0.f) { iFlagIndex |= 1<<l; } }

		// If the cube is entirely inside or outside of the surface,
		// there is no job to be done in this marching-cube cell.
		if( iFlagIndex == 0 || iFlagIndex == 255 ) { continue; }

		// If there are vertices which is inside the surface...
		// Which edges intersect the surface?
		// If i-th edge intersects the surface, mark '1' at i-th bit of 'iEdgeFlags' 
		const int iEdgeFlags = CubeEdgeFlag[iFlagIndex];

		// Find the intersection point of the surface with each edg
		for( int iEdge=0; iEdge<12; iEdge++ )
		{
			// If there is an intersection on this edg
			if( iEdgeFlags & (1<<iEdge) )
			{
				if( *vertOnEdge[iEdge] < 0 )
				{
					const int v0 = EdgeConnection[iEdge][0];
					const int v1 = EdgeConnection[iEdge][1];

					const int n0 = Node[v0];
					const int n1 = Node[v1];

					const float alpha = Abs( Phi[v0] / ( Phi[v1] - Phi[v0] ) );
					const float beta = 1 - alpha;

					pt.x = (VertexOffset[v0][0] + alpha * EdgeDirection[iEdge][0])/rx + basePos.x;
					pt.y = (VertexOffset[v0][1] + alpha * EdgeDirection[iEdge][1])/ry + basePos.y;
					pt.z = (VertexOffset[v0][2] + alpha * EdgeDirection[iEdge][2])/rz + basePos.z;

					positions.append( pt ); //push_back? == append?

					*vertOnEdge[iEdge] = vertCount++;
				}
			}
		}

		for( int iTriangle=0; iTriangle<5; iTriangle++ )
		{
			const int iTriangle3 = iTriangle * 3;

			// If there isn't any triangle to be drawn, escape this loop.
			if( TriangleConnectionTable[iFlagIndex][iTriangle3] < 0 ) { break; }

			for( int iCorner=0; iCorner<3; iCorner++ )
			{
				int iEdge = TriangleConnectionTable[iFlagIndex][iTriangle3+iCorner];

				threeEdges[iCorner] = iEdge;
				threeVerts[iCorner] = *vertOnEdge[iEdge];
			}

			polyConnections.append( threeVerts[0] );
			polyConnections.append( threeVerts[1] );
			polyConnections.append( threeVerts[2] );
		}
	}}}

	const int numVertices = positions.length();
	const int numTrifaces = (int)(polyConnections.length()/3);

	if( !numVertices || !numTrifaces ) { mesh.reset(); return true; }

	mesh.position.copyFrom(positions);
    mesh.indices.copyFrom(polyConnections);

	return true;
}

BORA_NAMESPACE_END

