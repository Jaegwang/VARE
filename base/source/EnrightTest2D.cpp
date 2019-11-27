//-------------------//
// EnrightTest2D.cpp //
//-------------------------------------------------------//
// author: Julie Jang @ Dexter Studios                   //
// last update: 2018.04.25                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

EnrightTest2D::EnrightTest2D()
{
}

void
EnrightTest2D::initialize( size_t gridRes )
{
	AABB2f aabb = AABB2f( Vec2f(0.f), Vec2f(1.f) );
	_grd = Grid2D( gridRes, aabb );

	_lvs.initialize( _grd, kUnified );
	_vel.initialize( _grd, kUnified );

	_setVelocityField(0.f);
	const float Lx = aabb.width(0);
	const Vec2f minPt = aabb.minPoint();
	Vec2f center = Vec2f(0.5f*Lx, 0.75f*Lx) + minPt;
	float radius = 0.15*Lx;
	SetCircle( center, radius, _lvs );
}

void
EnrightTest2D::setAdvectionConditions( float cfl_number, int minSubSteps, int maxSubSteps, float Dt, AdvectionScheme scheme )
{
	_adv.set( cfl_number, minSubSteps, maxSubSteps, Dt, scheme );
}

int 
EnrightTest2D::update( float t )
{
	_setVelocityField(t);
	return _adv.advect( _lvs, _vel );
}

void
EnrightTest2D::drawVelocityField() // in normal space
{
	if( _vel.numCells()==0 )  { return; } 

	int i, j;
	float x, y, h;

	AABB2f aabb = _vel.boundingBox();
	float L = aabb.width(0);
	Vec2f minPt = aabb.minPoint();

	int nx = _vel.nx();
	int ny = _vel.ny();
	float f = L/50.f/(float)nx;

	glColor3f ( 1.0f, 1.0f, 1.0f );
	glLineWidth ( 1.0f );
	glBegin ( GL_LINES );
	for ( j=0 ; j<ny ; j++ )
	for ( i=0 ; i<nx ; i++ )
	{
		Vec2f p = _vel.worldPoint( Vec2f(i,j) );
		Vec2f p0 = ((p+Vec2f(L/(float)nx/2.f)-minPt)/L)-Vec2f(0.5f);
		Vec2f vel = _vel(i,j);
		Vec2f q = (p+(f*vel));
		Vec2f q0 = ((q+Vec2f(L/(float)nx/2.f)-minPt)/L)-Vec2f(0.5f);
		glVertex2f ( p0.x, p0.y );
		glVertex2f ( q0.x, q0.y );
	}
	glEnd ();
}

void
EnrightTest2D::drawScalarField()
{
	AABB2f aabb = _lvs.boundingBox();
	float diagonalLength = aabb.diagonalLength();
	_lvs.drawLevelset( diagonalLength, diagonalLength );
}

void 
EnrightTest2D::_setVelocityField( float t )
{
	int nx = _vel.nx();
	int ny = _vel.ny();
	float _T = 8.f;
	float tr = Cos(PI*t/_T);
    //#pragma omp parallel for
	for( int j=0; j<ny; j++ )
	for( int i=0; i<nx; i++ )
	{
		int idx = _vel.cellIndex(i,j);
		Vec2f p = _vel.cellCenter(i,j)/(float)nx;
		Vec2f v;
		v.x = -Sin(2.f*PI*p.y)*Pow2(Sin(PI*p.x));
		v.y = Sin(2.f*PI*p.x)*Pow2(Sin(PI*p.y));
		_vel[idx] = tr*v*(float)nx;
	}
}

BORA_NAMESPACE_END

