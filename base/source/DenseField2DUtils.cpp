//---------------------//
// DenseField2DUtils.cpp //
//-------------------------------------------------------//
// author: Julie Jang @ Dexter Studios                   //
// last update: 2018.05.15                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

void SetCircle( const Vec2f& center/*worldspace*/, const float radius/*worldspace*/, ScalarDenseField2D& lvs/*voxelSpace*/ )
{
	const AABB2f aabb = lvs.boundingBox();
	const float Lx = aabb.width(0);
	const float Ly = aabb.width(1);

    const int nx = lvs.nx();
    const int ny = lvs.ny();

	const float w2n_x = (float)nx/Lx;
	const float w2n_y = (float)ny/Ly;
	const int wslargestDim = Max( Lx, Ly );
	const float rLx = (float)Lx/(float)wslargestDim;
	const float rLy = (float)Ly/(float)wslargestDim;


	const int vslargestDim = Max( nx, ny );
	const float rNx = (float)nx/(float)vslargestDim;
	const float rNy = (float)ny/(float)vslargestDim;

	float vsRadius = radius*Min(rNx*w2n_x, rNy*w2n_y);

//    #pragma omp parallel for
    for( int j=0; j<ny; j++ )
    for( int i=0; i<nx; i++ )
    {
        const int idx = lvs.cellIndex(i,j);
        const Vec2f p = lvs.worldPoint(Vec2f(i,j)+Vec2f(0.5));

		float wsDistToCenter = p.distanceTo( center );

		// convert world space distance to Interface to voxel space
        float dist = Sqrt(Pow2(w2n_x*(1.f/rLx)*rNx*(center.x-p.x))+Pow2(w2n_y*(1.f/rLy)*rNy*(center.y-p.y)));
		lvs[idx] = dist-vsRadius;
//		if( dist<vsRadius ) { lvs[idx] *= -1.f; }
    }
}

bool Gradient( VectorDenseField2D& v, const ScalarDenseField2D& s )
{
    return true;
}

bool Divergence( ScalarDenseField2D& s, const VectorDenseField2D& v )
{
    return true;
}

bool Curl( ScalarDenseField2D& s, const VectorDenseField2D& v )
{
    return true;
}

BORA_NAMESPACE_END

