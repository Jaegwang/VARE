//------------//
// FieldOP.cu //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.09.04                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

void
FieldOP::gradient( const ScalarDenseField& src, VectorDenseField& dst )
{
    //dst.initialize( src.getGrid() );
    size_t _I = src.nx()-1;
    size_t _J = src.ny()-1;
    size_t _K = src.nz()-1;

    float* pSrc = src.pointer();
    Vec3f* pDst = dst.pointer();

    auto kernel = [=] BORA_DEVICE ( size_t n )
    {
        size_t i,j,k;
        src.cellIndices( n, i,j,k );

        const float& v = pSrc[n];

        const float& v_i0 = i ==  0 ? 2.f*v-src(i+1,j,k) : src(i-1,j,k);
        const float& v_i1 = i == _I ? 2.f*v-src(i-1,j,k) : src(i+1,j,k);
        const float& v_j0 = j ==  0 ? 2.f*v-src(i,j+1,k) : src(i,j-1,k);
        const float& v_j1 = j == _J ? 2.f*v-src(i,j-1,k) : src(i,j+1,k);
        const float& v_k0 = k ==  0 ? 2.f*v-src(i,j,k+1) : src(i,j,k-1);
        const float& v_k1 = k == _K ? 2.f*v-src(i,j,k-1) : src(i,j,k+1);

        pDst[n] = Vec3f( (v_i1-v_i0)*0.5f,
                         (v_j1-v_j0)*0.5f,
                         (v_k1-v_k0)*0.5f );
    };

    LaunchCudaDevice( kernel, 0, src.size() );
    SyncCuda();
}

void
FieldOP::gradient( const ScalarSparseField& src, VectorSparseField& dst )
{
    size_t _I = src.nx()-1;
    size_t _J = src.ny()-1;
    size_t _K = src.nz()-1;

    float* pSrc = src.pointer();
    Vec3f* pDst = dst.pointer();

    auto kernel = [=] BORA_DEVICE ( size_t n )
    {
        size_t i,j,k;
        src.findIndices( n, i,j,k );

        const float& v = pSrc[n];

        const float& v_i0 = i ==  0 ? 2.f*v-src(i+1,j,k) : src(i-1,j,k);
        const float& v_i1 = i == _I ? 2.f*v-src(i-1,j,k) : src(i+1,j,k);
        const float& v_j0 = j ==  0 ? 2.f*v-src(i,j+1,k) : src(i,j-1,k);
        const float& v_j1 = j == _J ? 2.f*v-src(i,j-1,k) : src(i,j+1,k);
        const float& v_k0 = k ==  0 ? 2.f*v-src(i,j,k+1) : src(i,j,k-1);
        const float& v_k1 = k == _K ? 2.f*v-src(i,j,k-1) : src(i,j,k+1);

        pDst[n] = Vec3f( (v_i1-v_i0)*0.5f,
                         (v_j1-v_j0)*0.5f,
                         (v_k1-v_k0)*0.5f );
    };

    LaunchCudaDevice( kernel, 0, src.size() );
    SyncCuda();
}

void
FieldOP::normalGradient( const ScalarSparseField& src, VectorSparseField& dst )
{
    size_t _I = src.nx()-1;
    size_t _J = src.ny()-1;
    size_t _K = src.nz()-1;

    float* pSrc = src.pointer();
    Vec3f* pDst = dst.pointer();

    auto kernel = [=] BORA_DEVICE ( size_t n )
    {
        size_t i,j,k;
        src.findIndices( n, i,j,k );

        const float& v = pSrc[n];

        const float& v_i0 = i ==  0 ? 2.f*v-src(i+1,j,k) : src(i-1,j,k);
        const float& v_i1 = i == _I ? 2.f*v-src(i-1,j,k) : src(i+1,j,k);
        const float& v_j0 = j ==  0 ? 2.f*v-src(i,j+1,k) : src(i,j-1,k);
        const float& v_j1 = j == _J ? 2.f*v-src(i,j-1,k) : src(i,j+1,k);
        const float& v_k0 = k ==  0 ? 2.f*v-src(i,j,k+1) : src(i,j,k-1);
        const float& v_k1 = k == _K ? 2.f*v-src(i,j,k-1) : src(i,j,k+1);

        pDst[n] = Vec3f( (v_i1-v_i0)*0.5f,
                         (v_j1-v_j0)*0.5f,
                         (v_k1-v_k0)*0.5f ).normalized();
    };

    LaunchCudaDevice( kernel, 0, src.size() );
    SyncCuda();
}

void
FieldOP::divergence( const VectorDenseField& src, ScalarDenseField& dst )
{
    //dst.initialize( src.getGrid() );
    size_t _I = src.nx()-1;
    size_t _J = src.ny()-1;
    size_t _K = src.nz()-1;

    Vec3f* pSrc = src.pointer();
    float* pDst = dst.pointer();

    auto kernel = [=] BORA_DEVICE ( size_t n )
    {
        size_t i,j,k;
        src.cellIndices( n, i,j,k );

        const Vec3f& v = pSrc[n];

        const float& v_i0 = i ==  0 ? v.x*2.f-src(i+1,j,k).x : src(i-1,j,k).x;
        const float& v_i1 = i == _I ? v.x*2.f-src(i-1,j,k).x : src(i+1,j,k).x;
        const float& v_j0 = j ==  0 ? v.y*2.f-src(i,j+1,k).y : src(i,j-1,k).y;
        const float& v_j1 = j == _J ? v.y*2.f-src(i,j-1,k).y : src(i,j+1,k).y;
        const float& v_k0 = k ==  0 ? v.z*2.f-src(i,j,k+1).z : src(i,j,k-1).z;
        const float& v_k1 = k == _K ? v.z*2.f-src(i,j,k-1).z : src(i,j,k+1).z;

        pDst[n] = (v_i1-v_i0)*0.5f+
                  (v_j1-v_j0)*0.5f+ 
                  (v_k1-v_k0)*0.5f;
    };

    LaunchCudaDevice( kernel, 0, src.size() );
    SyncCuda();
}

void
FieldOP::divergence( const VectorSparseField& src, ScalarSparseField& dst )
{
    size_t _I = src.nx()-1;
    size_t _J = src.ny()-1;
    size_t _K = src.nz()-1;

    Vec3f* pSrc = src.pointer();
    float* pDst = dst.pointer();

    auto kernel = [=] BORA_DEVICE ( size_t n )
    {
        size_t i,j,k;
        src.findIndices( n, i,j,k );

        const Vec3f& v = pSrc[n];

        const float& v_i0 = i ==  0 ? v.x*2.f-src(i+1,j,k).x : src(i-1,j,k).x;
        const float& v_i1 = i == _I ? v.x*2.f-src(i-1,j,k).x : src(i+1,j,k).x;
        const float& v_j0 = j ==  0 ? v.y*2.f-src(i,j+1,k).y : src(i,j-1,k).y;
        const float& v_j1 = j == _J ? v.y*2.f-src(i,j-1,k).y : src(i,j+1,k).y;
        const float& v_k0 = k ==  0 ? v.z*2.f-src(i,j,k+1).z : src(i,j,k-1).z;
        const float& v_k1 = k == _K ? v.z*2.f-src(i,j,k-1).z : src(i,j,k+1).z;

        pDst[n] = (v_i1-v_i0)*0.5f+
                  (v_j1-v_j0)*0.5f+
                  (v_k1-v_k0)*0.5f;
    };

    LaunchCudaDevice( kernel, 0, src.size() );
    SyncCuda();
}

void
FieldOP::curvature( const ScalarDenseField& surf, ScalarDenseField& curv )
{
    size_t _I = surf.nx()-1;
    size_t _J = surf.ny()-1;
    size_t _K = surf.nz()-1;

    float* pSurf = surf.pointer();
    float* pCurv = curv.pointer();

    auto kernel = [=] BORA_DEVICE ( size_t n )
    {
        size_t i,j,k;
        surf.cellIndices( n, i,j,k );

        const float& s = pSurf[n];

        const float& s_i0 = i ==  0 ? s*2.f-surf(i+1,j,k) : surf(i-1,j,k);
        const float& s_i1 = i == _I ? s*2.f-surf(i-1,j,k) : surf(i+1,j,k);
        const float& s_j0 = j ==  0 ? s*2.f-surf(i,j+1,k) : surf(i,j-1,k);
        const float& s_j1 = j == _J ? s*2.f-surf(i,j-1,k) : surf(i,j+1,k);
        const float& s_k0 = k ==  0 ? s*2.f-surf(i,j,k+1) : surf(i,j,k-1);
        const float& s_k1 = k == _K ? s*2.f-surf(i,j,k-1) : surf(i,j,k+1);

        pCurv[n] = (6.f*s -s_i0-s_i1-s_j0-s_j1-s_k0-s_k1);
    };

    LaunchCudaDevice( kernel, 0, surf.size() );
    SyncCuda();
}

void
FieldOP::curvature( const ScalarSparseField& surf, ScalarSparseField& curv )
{
    size_t _I = surf.nx()-1;
    size_t _J = surf.ny()-1;
    size_t _K = surf.nz()-1;

    float* pSurf = surf.pointer();
    float* pCurv = curv.pointer();

    auto kernel = [=] BORA_DEVICE ( const size_t n )
    {
        size_t i,j,k;
        surf.findIndices( n, i,j,k );

        const float& s = pSurf[n];

        const float& s_i0 = i ==  0 ? s*2.f-surf(i+1,j,k) : surf(i-1,j,k);
        const float& s_i1 = i == _I ? s*2.f-surf(i-1,j,k) : surf(i+1,j,k);
        const float& s_j0 = j ==  0 ? s*2.f-surf(i,j+1,k) : surf(i,j-1,k);
        const float& s_j1 = j == _J ? s*2.f-surf(i,j-1,k) : surf(i,j+1,k);
        const float& s_k0 = k ==  0 ? s*2.f-surf(i,j,k+1) : surf(i,j,k-1);
        const float& s_k1 = k == _K ? s*2.f-surf(i,j,k-1) : surf(i,j,k+1);

        pCurv[n] = (6.f*s -s_i0-s_i1-s_j0-s_j1-s_k0-s_k1);
    };

    LaunchCudaDevice( kernel, 0, surf.size() );
    SyncCuda();
}

void
FieldOP::curvature( const VectorDenseField& norm, ScalarDenseField& curv )
{
    size_t EI = norm.nx()-1;
    size_t EJ = norm.ny()-1;
    size_t EK = norm.nz()-1;

    Vec3f* pNorm = norm.pointer();
    float* pCurv = curv.pointer();

    auto kernel = [=] BORA_DEVICE ( size_t n )
    {
        size_t i,j,k;
        norm.cellIndices( n, i,j,k );

        const Vec3f& s = pNorm[n];

        const float s_i0 = i ==  0 ? s.x*2.f-norm(i+1,j,k).x : norm(i-1,j,k).x;
        const float s_i1 = i == EI ? s.x*2.f-norm(i-1,j,k).x : norm(i+1,j,k).x;
        const float s_j0 = j ==  0 ? s.y*2.f-norm(i,j+1,k).y : norm(i,j-1,k).y;
        const float s_j1 = j == EJ ? s.y*2.f-norm(i,j-1,k).y : norm(i,j+1,k).y;
        const float s_k0 = k ==  0 ? s.z*2.f-norm(i,j,k+1).z : norm(i,j,k-1).z;
        const float s_k1 = k == EK ? s.z*2.f-norm(i,j,k-1).z : norm(i,j,k+1).z;

        pCurv[n] = (s_i1-s_i0)*0.5f + (s_j1-s_j0)*0.5f + (s_k1-s_k0)*0.5f;
    };

    LaunchCudaDevice( kernel, 0, norm.size() );
    SyncCuda();
}

void
FieldOP::curvature( const VectorSparseField& norm, ScalarSparseField& curv )
{
    size_t EI = norm.nx()-1;
    size_t EJ = norm.ny()-1;
    size_t EK = norm.nz()-1;

    Vec3f* pNorm = norm.pointer();
    float* pCurv = curv.pointer();

    auto kernel = [=] BORA_DEVICE ( size_t n )
    {
        size_t i,j,k;
        norm.findIndices( n, i,j,k );

        const Vec3f& s = pNorm[n];

        const float s_i0 = i ==  0 ? s.x*2.f-norm(i+1,j,k).x : norm(i-1,j,k).x;
        const float s_i1 = i == EI ? s.x*2.f-norm(i-1,j,k).x : norm(i+1,j,k).x;
        const float s_j0 = j ==  0 ? s.y*2.f-norm(i,j+1,k).y : norm(i,j-1,k).y;
        const float s_j1 = j == EJ ? s.y*2.f-norm(i,j-1,k).y : norm(i,j+1,k).y;
        const float s_k0 = k ==  0 ? s.z*2.f-norm(i,j,k+1).z : norm(i,j,k-1).z;
        const float s_k1 = k == EK ? s.z*2.f-norm(i,j,k-1).z : norm(i,j,k+1).z;

        pCurv[n] = (s_i1-s_i0)*0.5f + (s_j1-s_j0)*0.5f + (s_k1-s_k0)*0.5f;
    };

    LaunchCudaDevice( kernel, 0, norm.size() );
    SyncCuda();
}

void
FieldOP::curl( const VectorDenseField& vel, VectorDenseField& cur )
{
    size_t _I = vel.nx()-1;
    size_t _J = vel.ny()-1;
    size_t _K = vel.nz()-1;

    Vec3f* pCur = cur.pointer();

    auto kernel = [=] BORA_DEVICE ( const size_t n )
    {
        size_t i,j,k;
        vel.cellIndices( n, i,j,k );

        const Vec3f& v = vel[n];

        const Vec3f& v_x0 = i ==  0 ? v*2.f-vel(i+1,j,k) : vel(i-1,j,k);
        const Vec3f& v_x1 = i == _I ? v*2.f-vel(i-1,j,k) : vel(i+1,j,k);
        const Vec3f& v_y0 = j ==  0 ? v*2.f-vel(i,j+1,k) : vel(i,j-1,k);
        const Vec3f& v_y1 = j == _J ? v*2.f-vel(i,j-1,k) : vel(i,j+1,k);
        const Vec3f& v_z0 = k ==  0 ? v*2.f-vel(i,j,k+1) : vel(i,j,k-1);
        const Vec3f& v_z1 = k == _K ? v*2.f-vel(i,j,k-1) : vel(i,j,k+1);
        
        const float x = (v_y1.z-v_y0.z)*0.5f - (v_z1.y-v_z0.y)*0.5f;
        const float y = (v_z1.x-v_z0.x)*0.5f - (v_x1.z-v_x0.z)*0.5f;
        const float z = (v_x1.y-v_x0.y)*0.5f - (v_y1.x-v_y0.x)*0.5f;

        pCur[n] = Vec3f( x,y,z );
    };

    LaunchCudaDevice( kernel, 0, cur.size() );
    SyncCuda();
}

void
FieldOP::curl( const VectorSparseField& vel, VectorSparseField& cur )
{
    size_t _I = vel.nx()-1;
    size_t _J = vel.ny()-1;
    size_t _K = vel.nz()-1;

    Vec3f* pCur = cur.pointer();

    auto kernel = [=] BORA_DEVICE ( const size_t n )
    {
        size_t i,j,k;
        vel.findIndices( n, i,j,k );

        const Vec3f& v = vel[n];

        const Vec3f& v_x0 = i ==  0 ? v*2.f-vel(i+1,j,k) : vel(i-1,j,k);
        const Vec3f& v_x1 = i == _I ? v*2.f-vel(i-1,j,k) : vel(i+1,j,k);
        const Vec3f& v_y0 = j ==  0 ? v*2.f-vel(i,j+1,k) : vel(i,j-1,k);
        const Vec3f& v_y1 = j == _J ? v*2.f-vel(i,j-1,k) : vel(i,j+1,k);
        const Vec3f& v_z0 = k ==  0 ? v*2.f-vel(i,j,k+1) : vel(i,j,k-1);
        const Vec3f& v_z1 = k == _K ? v*2.f-vel(i,j,k-1) : vel(i,j,k+1);
        
        const float x = (v_y1.z-v_y0.z)*0.5f - (v_z1.y-v_z0.y)*0.5f;
        const float y = (v_z1.x-v_z0.x)*0.5f - (v_x1.z-v_x0.z)*0.5f;
        const float z = (v_x1.y-v_x0.y)*0.5f - (v_y1.x-v_y0.x)*0.5f;

        pCur[n] = Vec3f( x,y,z );
    };

    LaunchCudaDevice( kernel, 0, cur.size() );
    SyncCuda();
}

void
FieldOP::normalCurl( const VectorSparseField& vel, VectorSparseField& cur )
{
    size_t _I = vel.nx()-1;
    size_t _J = vel.ny()-1;
    size_t _K = vel.nz()-1;

    Vec3f* pCur = cur.pointer();

    auto kernel = [=] BORA_DEVICE ( const size_t n )
    {
        size_t i,j,k;
        vel.findIndices( n, i,j,k );

        const Vec3f& v = vel[n];

        const Vec3f v_x0 = (i ==  0 ? v*2.f-vel(i+1,j,k) : vel(i-1,j,k)).normalized();
        const Vec3f v_x1 = (i == _I ? v*2.f-vel(i-1,j,k) : vel(i+1,j,k)).normalized();
        const Vec3f v_y0 = (j ==  0 ? v*2.f-vel(i,j+1,k) : vel(i,j-1,k)).normalized();
        const Vec3f v_y1 = (j == _J ? v*2.f-vel(i,j-1,k) : vel(i,j+1,k)).normalized();
        const Vec3f v_z0 = (k ==  0 ? v*2.f-vel(i,j,k+1) : vel(i,j,k-1)).normalized();
        const Vec3f v_z1 = (k == _K ? v*2.f-vel(i,j,k-1) : vel(i,j,k+1)).normalized();
        
        const float x = (v_y1.z-v_y0.z)*0.5f - (v_z1.y-v_z0.y)*0.5f;
        const float y = (v_z1.x-v_z0.x)*0.5f - (v_x1.z-v_x0.z)*0.5f;
        const float z = (v_x1.y-v_x0.y)*0.5f - (v_y1.x-v_y0.x)*0.5f;

        pCur[n] = Vec3f( x,y,z );
    };

    LaunchCudaDevice( kernel, 0, cur.size() );
    SyncCuda();
}

BORA_NAMESPACE_END

