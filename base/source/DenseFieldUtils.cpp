//---------------------//
// DenseFieldUtils.cpp //
//-------------------------------------------------------//
// author: Julie Jang @ Dexter Studios                   //
// last update: 2018.04.12                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN


//TODO: TEST ALL CASES
void SetSphere( const Vec3f& center, const float radius, ScalarDenseField& lvs )
{
	const AABB3f aabb = lvs.boundingBox();
	const float Lx = aabb.width(0);
	const float Ly = aabb.width(1);
	const float Lz = aabb.width(2);

    const int nx = lvs.nx();
    const int ny = lvs.ny();
    const int nz = lvs.nz();

	const float w2n_x = (float)nx/Lx;
	const float w2n_y = (float)ny/Ly;
	const float w2n_z = (float)nz/Lz;

	const int largestDim = Max( nx, ny, nz );

	const float r1 = (float)nx/(float)largestDim;
	const float r2 = (float)ny/(float)largestDim;
	const float r3 = (float)nz/(float)largestDim;

	const float vsRadius = radius*Min(r1*w2n_x,r2*w2n_y,r3*w2n_z);

    //#pragma omp parallel for
    for( int k=0; k<nz; k++ )
    for( int j=0; j<ny; j++ )
    for( int i=0; i<nx; i++ )
    {
        const int idx = lvs.cellIndex(i,j,k);
        const Vec3f p = lvs.worldPoint(Vec3f(i,j,k)+Vec3f(0.5));

		float wsDistToCenter = p.distanceTo( center );

		// convert world space distance to Interface to voxel space
        float dist = Sqrt(Pow2(w2n_x*r1*(p.x-center.x))+Pow2(w2n_y*r2*(p.y-center.y))+Pow2(w2n_z*r3*(p.z-center.z)));
		lvs[idx] = dist-vsRadius;
		//if( dist<vsRadius ) { lvs[idx] *= -1.f; }
    }
}

bool Gradient( const ScalarDenseField& lvs, VectorDenseField& nrm )
{
    if( (Grid)lvs != (Grid)nrm )
    {
        COUT << "Error@CalcNormals(): Grid resolution mismatch." << ENDL;
        return false;
    }

    const int nx = nrm.nx();
    const int ny = nrm.ny();
    const int nz = nrm.nz();

    //#pragma omp parallel for
    for( int k=0; k<nz; k++ )
    for( int j=0; j<ny; j++ )
    for( int i=0; i<nx; i++ )
    {
        const int idx = nrm.cellIndex(i,j,k);

        const int i0 = (i==0   ) ? idx : lvs.i0(idx); // west
        const int i1 = (i==nx-1) ? idx : lvs.i1(idx); // east
        const int j0 = (j==0   ) ? idx : lvs.j0(idx); // south
        const int j1 = (j==ny-1) ? idx : lvs.j1(idx); // north
        const int k0 = (k==0   ) ? idx : lvs.k0(idx); // top
        const int k1 = (k==nz-1) ? idx : lvs.k1(idx); // bottom

        const float hx = ((i0==idx)||(i1==idx)) ? 1.f : 2.f;
        const float hy = ((j0==idx)||(j1==idx)) ? 1.f : 2.f;
        const float hz = ((k0==idx)||(k1==idx)) ? 1.f : 2.f;

        Vec3f& N = nrm[idx];

        N.x = ( lvs[i1] - lvs[i0] ) / hx;
        N.y = ( lvs[j1] - lvs[j0] ) / hy;
        N.z = ( lvs[k1] - lvs[k0] ) / hz;

        N.normalize();
    }

    return true;
}

BORA_NAMESPACE_END

