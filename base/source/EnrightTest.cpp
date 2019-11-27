//----------------//
// EnrightTest.cpp //
//-------------------------------------------------------//
// author: Julie Jang @ Dexter Studios                   //
// last update: 2018.04.10                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

EnrightTest::EnrightTest()
{
}

void
EnrightTest::initialize( int gridRes )
{
	_initialize( gridRes );
	TriangleMesh triMesh;
	AABB3f aabb = _lvs.boundingBox();
	const float Lx = aabb.width(0);
	const Vec3f minPt = aabb.minPoint();
	Vec3f center = Vec3f(0.35*Lx) + minPt;
	float radius = 0.15*Lx;
	SetSphere( center, radius, _lvs );
}

void
EnrightTest::initialize( int gridRes, std::string& filePath )
{
	TriangleMesh triMesh;
	if( filePath == "" || !triMesh.load( filePath.c_str() )) 
	{
		initialize( gridRes );
	}
	else
	{
		_initialize( gridRes );
		MarkerDenseField stt;
		Voxelizer_FMM lvsSolver;
		_lvs.initialize( _grd, kUnified );
		stt.setGrid( _grd );
		lvsSolver.set( _grd );
		lvsSolver.addMesh( _lvs, stt, triMesh, INFINITE, INFINITE );
		lvsSolver.finalize();
	}
}

void
EnrightTest::setAdvectionConditions( float cfl_number, int minSubSteps, int maxSubSteps, float Dt, AdvectionScheme scheme )
{
	_adv.set( cfl_number, minSubSteps, maxSubSteps, Dt, scheme );
}

int
EnrightTest::update( float t )
{
	_setVelocityField(t);
	return _adv.advect( _lvs, _vel );
}

void
EnrightTest::exportCache( std::string& filePath )
{
	_outputTriMesh.reset();
	_surfacer.reset();
	_surfacer.compute( _lvs, _outputTriMesh );
	_outputTriMesh.updateBoundingBox();
	_outputTriMesh.save( filePath.c_str() ); 
}

void
EnrightTest::drawWireframe()
{
	if( !_outputTriMesh.numTriangles() ) { return; }
	glPushAttrib( GL_ALL_ATTRIB_BITS );
	_outputTriMesh.drawWireframe();
	glPopAttrib();
}

void
EnrightTest::drawVelocityField( bool isDraw2D, float& scale, Vec3f& sliceRatio )
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
EnrightTest::drawScalarField()
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
			//Vec3f p = (_lvs.worldPoint(Vec3f(i,j,k))+Vec3f(L/_lvs.nx()/2.f)-minPt)/L;
			glVertex( p.x, p.y, p.z );
		}
	}
	glEnd();
}

void
EnrightTest::_initialize( int gridRes )
{
	AABB3f aabb = AABB3f( Vec3f(-50.f), Vec3f(50.f) );
	_grd = Grid( gridRes, aabb );
	_lvs.initialize( _grd, kUnified );
	_vel.initialize( _grd, kUnified );
	_setVelocityField(0.f);
}

void
EnrightTest::_setVelocityField( float t )
{
	int nx = _vel.nx();
	int ny = _vel.ny();
	int nz = _vel.nz();
	float _T = 3.f;
	float tr = Cos(PI*t/_T);
    //#pragma omp parallel for
	for( int k=0; k<nz; k++ )
	for( int j=0; j<ny; j++ )
	for( int i=0; i<nx; i++ )
	{
		int idx = _vel.cellIndex(i,j,k);
		Vec3f p = _vel.cellCenter(i,j,k)/(float)nx;
		Vec3f v;
		v.x = 2.f*Pow2(Sin(PI*p.x))*Sin(2.f*PI*p.y)*Sin(2.f*PI*p.z);
		v.y = -Sin(2.f*PI*p.x)*Pow2(Sin(PI*p.y))*Sin(2.f*PI*p.z);
		v.z = -Sin(2.f*PI*p.x)*Sin(2.f*PI*p.y)*Pow2(Sin(PI*p.z));
		_vel[idx] = tr*v*(float)nx;
	}
}

BORA_NAMESPACE_END

