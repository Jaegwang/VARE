//----------------//
// ZalesakTest.cpp //
//-------------------------------------------------------//
// author: Julie Jang @ Dexter Studios                   //
// last update: 2018.04.10                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

ZalesakTest::ZalesakTest()
{
}

void
ZalesakTest::initialize( const float gridRes )
{
	TriangleMesh triMesh;
	Vec3fArray& pos = triMesh.pos;
	IndexArray& ind = triMesh.indices;
	for( int i=0; i<369; ++i )
	{
		pos.append( ZalesakSphereVertices[i] );
	}
	for( int i=0; i<2202; ++i )
	{
		ind.append( ZalesakSphereIndices[i] );
	}

	AABB3f aabb = AABB3f( Vec3f(-50.f), Vec3f(50.f) );
	_grid = Grid( gridRes, aabb );

	MarkerDenseField stt;
	Voxelizer_FMM lvsSolver;
	_lvs.initialize( _grid, kUnified );
	stt.setGrid( _grid );
	lvsSolver.set( _grid );
	lvsSolver.addMesh( _lvs, stt, triMesh, INFINITE, INFINITE );
	lvsSolver.finalize();

	_vel.initialize( _grid, kUnified );
	_setVorticityField();
}

void
ZalesakTest::setAdvectionConditions( float cfl_number, int minSubSteps, int maxSubSteps, float Dt, AdvectionScheme scheme )
{
	_adv.set( cfl_number, minSubSteps, maxSubSteps, Dt, scheme );
}

int
ZalesakTest::update( const float t )
{
	return _adv.advect( _lvs, _vel );
}

void
ZalesakTest::exportCache( std::string& filePath )
{
	_outputTriMesh.reset();
	_surfacer.reset();
	_surfacer.compute( _lvs, _outputTriMesh );
	_outputTriMesh.updateBoundingBox();
	_outputTriMesh.save( filePath.c_str() ); 
}

void
ZalesakTest::drawWireframe()
{
	if( !_outputTriMesh.numTriangles() ) { return; }
	glPushAttrib( GL_ALL_ATTRIB_BITS );
	_outputTriMesh.drawWireframe();
	glPopAttrib();
}

void
ZalesakTest::drawVelocityField( bool isDraw2D, float& scale, Vec3f& sliceRatio )
{
	if( _vel.numCells()==0 )  { return; } 

	int i, j, k;
	float x, y, z, h;

	AABB3f aabb = _vel.boundingBox();
	float N = aabb.width(0);  

	int nx = _vel.nx();
	int ny = _vel.ny();
	int nz = _vel.nz();
	float f = 1.f*scale;

	glColor3f ( 1.0f, 1.0f, 1.0f );
	glLineWidth ( 1.0f );

	glBegin ( GL_LINES );

		if( !isDraw2D )
		{
			for ( i=0 ; i<nx ; i++ )
			for ( j=0 ; j<ny ; j++ )
			for ( k=0 ; k<nz ; k++ )
			{
				Vec3f p = _vel.worldPoint( Vec3f(i,j,k) );
				Vec3f vel = _vel(i,j,k);
				glVertex3f ( p.x, p.y, p.z );
				glVertex3f ( p.x+f*vel.x, p.y+f*vel.y, p.z+f*vel.z );
			}
		}
		else
		{
			//draw xSlice
			i = Clamp( (int) (sliceRatio.x*(nx-1)), 0, nx-1 );
			for ( j=0 ; j<ny ; j++ )
			for ( k=0 ; k<nz ; k++ )
			{
				Vec3f p = _vel.worldPoint( Vec3f(i,j,k) );
				Vec3f vel = _vel(i,j,k);
				glVertex3f ( p.x, p.y, p.z );
				glVertex3f ( p.x+f*vel.x, p.y+f*vel.y, p.z+f*vel.z );
			}

			//draw ySlice
			j = Clamp( (int) (sliceRatio.y*(ny-1)), 0, ny-1 );
			for ( i=0 ; i<nx ; i++ )
			for ( k=0 ; k<nz ; k++ )
			{
				Vec3f p = _vel.worldPoint( Vec3f(i,j,k) );
				Vec3f vel = _vel(i,j,k);
				glVertex3f ( p.x, p.y, p.z );
				glVertex3f ( p.x+f*vel.x, p.y+f*vel.y, p.z+f*vel.z );
			}

			//draw zSlice
			k = Clamp( (int) (sliceRatio.z*(nz-1)), 0, nz-1 );
			for ( i=0 ; i<nx ; i++ )
			for ( j=0 ; j<ny ; j++ )
			{
				Vec3f p = _vel.worldPoint( Vec3f(i,j,k) );
				Vec3f vel = _vel(i,j,k);
				glVertex3f ( p.x, p.y, p.z );
				glVertex3f ( p.x+f*vel.x, p.y+f*vel.y, p.z+f*vel.z );
			}
		}

	glEnd ();
}

void
ZalesakTest::drawScalarField()
{
	int nx = _lvs.nx();
	int ny = _lvs.ny();
	int nz = _lvs.nz();
	float minValue = _lvs.minValue();
	AABB3f aabb = _vel.boundingBox();
	float L = aabb.width(0);
	Vec3f minPt = aabb.minPoint();

	glPointSize( 5 );
	glBegin( GL_POINTS );
	for( int k=0; k<nz; k++ )
	for( int j=0; j<ny; j++ )
	for( int i=0; i<nx; i++ )
	{
		float value = _lvs( i, j, k );
		if( value<EPSILON )
		{
			float normalizedValue = value/minValue;
			glColor( normalizedValue, 0.f, 0.f );
			Vec3f p = _lvs.worldPoint( Vec3f(i, j, k) ) + Vec3f(L/(float)nx/2.f);
			glVertex( p.x, p.y, p.z );
		}
	}
	glEnd();
}

void
ZalesakTest::_setVorticityField()
{
	int nx = _vel.nx();
	int ny = _vel.ny();
	int nz = _vel.nz();
    //#pragma omp parallel for
	for( int k=0; k<nz; k++ )
	for( int j=0; j<ny; j++ )
	for( int i=0; i<nx; i++ )
	{
		int idx = _vel.cellIndex(i,j,k);
		Vec3f p = _vel.cellCenter(i,j,k)/(float)nx;
		Vec3f v;
		v.x = (PI/314.f)*(.5f-p.y);
		v.y = (PI/314.f)*(p.x-.5f);
		v.z = 0.f;
		_vel[idx] = v*(float)nx*100.f;
	}
}

BORA_NAMESPACE_END

