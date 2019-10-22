
#include <VARE.h>

VARE_NAMESPACE_BEGIN

void Rasterization::
PointsToVelField( ScalarDenseField& mass, VectorDenseField& velo, const Vec3fArray& p_pos, const Vec3fArray& p_vel )
{
    mass.zeroize();
    velo.zeroize();

    int EI(velo.nx()-1), EJ(velo.ny()-1), EK(velo.nz()-1);

    float* pMass = mass.pointer();
    Vec3f* pVelo = velo.pointer();

    auto kernel = VARE_DEVICE_KERNEL
    {
        const Vec3f& p = p_pos[ix];
        const Vec3f& v = p_vel[ix];

        if( mass.inside( p ) == false ) return;

        int ci = Max( (int)(p.x-0.5f), 0 );
        int cj = Max( (int)(p.y-0.5f), 0 );
        int ck = Max( (int)(p.z-0.5f), 0 );

        for( int k=ck; k<=Min(ck+1,EK); ++k )
        for( int j=cj; j<=Min(cj+1,EJ); ++j )
        for( int i=ci; i<=Min(ci+1,EI); ++i )
        {
            Vec3f cell = mass.cellCenter( i,j,k );

            const Vec3f dev = p - cell;
            // Tri-linear weight.
            const float xw = Max( 0.f, 1.f-Abs(dev.x) );
            const float yw = Max( 0.f, 1.f-Abs(dev.y) );
            const float zw = Max( 0.f, 1.f-Abs(dev.z) );
            const float w = xw*yw*zw + 1e-05f;

            size_t idx = mass.cellIndex( i,j,k );
            
            float& accw = pMass[idx];
            float& accx = pVelo[idx].x;
            float& accy = pVelo[idx].y;
            float& accz = pVelo[idx].z;

            atomicAdd( &accw, w );
            atomicAdd( &accx, w*v.x );
            atomicAdd( &accy, w*v.y );
            atomicAdd( &accz, w*v.z );
        }
    };

    LaunchDeviceKernel( kernel, 0, p_pos.size() );
    SyncKernels();

    auto kernel2 = VARE_DEVICE_KERNEL
    {
        if( pMass[ix] > 0.f )
        {
            pVelo[ix] /= pMass[ix];
        }
        else
        {
            pMass[ix] = 0.f;
            pVelo[ix] = Vec3f(0.f);
        }
    };

    LaunchDeviceKernel( kernel2, 0, velo.size() );
    SyncKernels();
}

void Rasterization::
ParticlesToVelField( ScalarDenseField& mass, VectorDenseField& velo, const Particles& pts )
{
    PointsToVelField( mass, velo, pts.position, pts.velocity );
}

void Rasterization::
PointsToCountField( ScalarDenseField& mass, const Vec3fArray& p_pos )
{
    mass.zeroize();
    int EI(mass.nx()-1), EJ(mass.ny()-1), EK(mass.nz()-1);

    float* pMass = mass.pointer();

    auto kernel = VARE_DEVICE_KERNEL
    {
        const Vec3f& p = p_pos[ix];
        if( mass.inside( p ) == false ) return;

        int ci = Max( (int)(p.x-0.5f), 0 );
        int cj = Max( (int)(p.y-0.5f), 0 );
        int ck = Max( (int)(p.z-0.5f), 0 );

        for( int k=ck; k<=Min(ck+1,EK); ++k )
        for( int j=cj; j<=Min(cj+1,EJ); ++j )
        for( int i=ci; i<=Min(ci+1,EI); ++i )
        {
            Vec3f cell = mass.cellCenter( i,j,k );

            size_t idx = mass.cellIndex( i,j,k );
            
            float& accw = pMass[idx];
            atomicAdd( &accw, 1.f );
        }
    };

    LaunchDeviceKernel( kernel, 0, p_pos.size() );
    SyncKernels();
}

void Rasterization::
PointsToVelField( ScalarSparseField& mass, VectorSparseField& velo, const Vec3fArray& p_pos, const Vec3fArray& p_vel )
{
    mass.zeroize();
    velo.zeroize();

    int EI(velo.nx()-1), EJ(velo.ny()-1), EK(velo.nz()-1);

    float* pMass = mass.pointer();
    Vec3f* pVelo = velo.pointer();

    auto kernel = VARE_DEVICE_KERNEL
    {
        const Vec3f& p = p_pos[ix];
        const Vec3f& v = p_vel[ix];

        if( mass.inside( p ) == false ) return;

        int ci = Max( (int)(p.x-0.5f), 0 );
        int cj = Max( (int)(p.y-0.5f), 0 );
        int ck = Max( (int)(p.z-0.5f), 0 );

        for( int k=ck; k<=Min(ck+1,EK); ++k )
        for( int j=cj; j<=Min(cj+1,EJ); ++j )
        for( int i=ci; i<=Min(ci+1,EI); ++i )
        {
            if( mass.isWritable(i,j,k) == false ) continue;

            Vec3f cell = mass.cellCenter( i,j,k );

            const Vec3f dev = p - cell;
            // Tri-linear weight.
            const float xw = Max( 0.f, 1.f-Abs(dev.x) );
            const float yw = Max( 0.f, 1.f-Abs(dev.y) );
            const float zw = Max( 0.f, 1.f-Abs(dev.z) );
            const float w = xw*yw*zw + 1e-05f;

            size_t idx = mass.findIndex( i,j,k );
            
            float& accw = pMass[idx];
            float& accx = pVelo[idx].x;
            float& accy = pVelo[idx].y;
            float& accz = pVelo[idx].z;

            atomicAdd( &accw, w );
            atomicAdd( &accx, w*v.x );
            atomicAdd( &accy, w*v.y );
            atomicAdd( &accz, w*v.z );
        }
    };

    LaunchDeviceKernel( kernel, 0, p_pos.size() );
    SyncKernels();

    auto kernel2 = VARE_DEVICE_KERNEL
    {
        if( pMass[ix] > 0.f )
        {
            pVelo[ix] /= pMass[ix];
        }
        else
        {
            pMass[ix] = 0.f;
            pVelo[ix] = Vec3f(0.f);
        }
    };

    LaunchDeviceKernel( kernel2, 0, velo.size() );
    SyncKernels();
}

void Rasterization::
ParticlesToVelField( ScalarSparseField& mass, VectorSparseField& velo, const Particles& pts )
{
    PointsToVelField( mass, velo, pts.position, pts.velocity );
}

void Rasterization::
PointsToCountField( ScalarSparseField& mass, const PointArray& p_pos )
{
    mass.zeroize();

    int EI(mass.nx()-1), EJ(mass.ny()-1), EK(mass.nz()-1);
    float* pMass = mass.pointer();

    auto kernel = VARE_DEVICE_KERNEL
    {
        const Vec3f& p = p_pos[ix];
        if( mass.inside( p ) == false ) return;

        int ci = Max( (int)(p.x-0.5f), 0 );
        int cj = Max( (int)(p.y-0.5f), 0 );
        int ck = Max( (int)(p.z-0.5f), 0 );

        for( int k=ck; k<=Min(ck+1,EK); ++k )
        for( int j=cj; j<=Min(cj+1,EJ); ++j )
        for( int i=ci; i<=Min(ci+1,EI); ++i )
        {
            if( mass.isWritable(i,j,k) == false ) continue;

            size_t idx = mass.findIndex( i,j,k );
            
            float& accw = pMass[idx];
            atomicAdd( &accw, 1.f );
        }
    };

    LaunchDeviceKernel( kernel, 0, p_pos.size() );
    SyncKernels();
}


void Rasterization::
ParticlesToSurface( ScalarSparseField& surf, VectorSparseField& pose, const Particles& pts, const float influence, const float droplet )
{
    surf.background = 0.f;
    surf.zeroize();

    pose.zeroize();

    float* pSurf = surf.pointer();
    Vec3f* pPose = pose.pointer();

    const float h = influence;
    const float r = droplet;

    auto kernel = VARE_DEVICE_KERNEL
    {
        const Vec3f& p = pts.position[ix];

        if( surf.inside( p ) == false ) return;

        Idx3 min, max;
        surf.neighborCells( p, min, max, (int)h );

        for( size_t k=min.k; k<=max.k; ++k )
        for( size_t j=min.j; j<=max.j; ++j )
        for( size_t i=min.i; i<=max.i; ++i )
        {        
             Vec3f cell = surf.cellCenter( i,j,k );

            const float dist = (p - cell).length();
            const float s = dist/h;

            float w = Max( 0.f, Pow3( 1.f-s*s ) );

            size_t idx = surf.findIndex( i,j,k );
            if( idx != INVALID_MAX )
            {
                float& accw = pSurf[idx];
                float& accx = pPose[idx].x;
                float& accy = pPose[idx].y;
                float& accz = pPose[idx].z;

                atomicAdd( &accw, w );
                atomicAdd( &accx, w*p.x );
                atomicAdd( &accy, w*p.y );
                atomicAdd( &accz, w*p.z );
            }
        }
    };

    LaunchDeviceKernel( kernel, 0, pts.position.size() );
    SyncKernels();

    surf.background = 10.f;

    auto kernel2 = VARE_DEVICE_KERNEL
    {
        size_t i,j,k;
        surf.findIndices( ix, i,j,k );

        if( pSurf[ix] > 1e-30f )
        {
            const Idx3 idx = surf.findIndices( ix );
            const Vec3f cell = surf.cellCenter( idx );

            pPose[ix] /= pSurf[ix];
            pSurf[ix] = pPose[ix].distanceTo( cell ) - r;
        }
        else
        {
            pSurf[ix] = 5.f;
        }
    };

    LaunchDeviceKernel( kernel2, 0, surf.size() );
    SyncKernels();
}

void Rasterization::
ParticlesToSurfaceAndVelocity( ScalarSparseField& surf, ScalarSparseField& mass, VectorSparseField& pose, VectorSparseField& velo, const Particles& pts, const float influence, const float droplet )
{
    surf.background = 0.f;
    surf.zeroize();

    mass.background = 0.f;
    mass.zeroize();

    pose.zeroize();
    velo.zeroize();

    float* pSurf = surf.pointer();
    float* pMass = mass.pointer();
    Vec3f* pPose = pose.pointer();
    Vec3f* pVelo = velo.pointer();

    const float h = influence;
    const float r = droplet;

    auto kernel = VARE_DEVICE_KERNEL
    {
        const Vec3f& p = pts.position[ix];
        const Vec3f& v = pts.velocity[ix];

        if( surf.inside( p ) == false ) return;

        Idx3 min, max;
        surf.neighborCells( p, min, max, (int)h );

        for( size_t k=min.k; k<=max.k; ++k )
        for( size_t j=min.j; j<=max.j; ++j )
        for( size_t i=min.i; i<=max.i; ++i )
        {
             Vec3f cell = surf.cellCenter( i,j,k );

            const float dist = (p - cell).length();
            const float s = dist/h;

            float w = Max( 0.f, Pow3( 1.f-s*s ) );

            size_t idx = surf.findIndex( i,j,k );
            if( idx != INVALID_MAX )
            {
                float& velm = pMass[idx];
                float& velx = pVelo[idx].x;
                float& vely = pVelo[idx].y;
                float& velz = pVelo[idx].z;

                atomicAdd( &velm, w );
                atomicAdd( &velx, w*v.x );
                atomicAdd( &vely, w*v.y );
                atomicAdd( &velz, w*v.z );
            }
       
            if( idx != INVALID_MAX )
            {
                float& accw = pSurf[idx];
                float& accx = pPose[idx].x;
                float& accy = pPose[idx].y;
                float& accz = pPose[idx].z;

                atomicAdd( &accw, w );
                atomicAdd( &accx, w*p.x );
                atomicAdd( &accy, w*p.y );
                atomicAdd( &accz, w*p.z );
            }
        }
    };

    LaunchDeviceKernel( kernel, 0, pts.position.size() );
    SyncKernels();

    surf.background = 10.f;

    auto kernel2 = VARE_DEVICE_KERNEL
    {
        if( pSurf[ix] > 1e-30f )
        {
            const Idx3 idx = surf.findIndices( ix );
            const Vec3f cell = surf.cellCenter( idx );

            pPose[ix] /= pSurf[ix];
            pSurf[ix] = pPose[ix].distanceTo( cell ) - r;
        }
        else
        {
            pSurf[ix] = 5.f;
        }

        if( pMass[ix] > 1e-30f )
        {
            pVelo[ix] /= pMass[ix];
        }
        else
        {
            pVelo[ix] = Vec3f(0.f);
        }
    };

    LaunchDeviceKernel( kernel2, 0, surf.size() );
    SyncKernels();
}

void Rasterization::
ReinitializeSurface( FloatSparseField& surf, FloatSparseField& temp, const int iter, const bool watertank, const float waterlevel )
{
    size_t _I(surf.nx()-1), _J(surf.ny()-1), _K(surf.nz()-1);
    const float band = (float)(_I + _J + _K);
    const float dt = 0.3f;
    
    surf.background = 10.0;
    temp.background = 10.0;

    for( int x=0; x<iter; ++x )
    {
        float* pSurf = surf.pointer();
        float* pTemp = temp.pointer();

        auto kernel = VARE_DEVICE_KERNEL
        {
            size_t i,j,k;
            surf.findIndices( ix, i,j,k );

            const float& v = pSurf[ix];

            const float sign = v / sqrt( v*v + 1.f );
            float v_i0, v_i1, v_j0, v_j1, v_k0, v_k1;

            if( watertank == true )
            {
                const Vec3f cell = surf.cellCenter( i,j,k );
                const float ll = cell.y-waterlevel;

                v_j0 = j ==  0 ? ll-1.f : surf(i,j-1,k);
                v_j1 = j == _J ? ll+1.f : surf(i,j+1,k);

                v_i0 = i ==  0 ? ll : surf(i-1,j,k);
                v_i1 = i == _I ? ll : surf(i+1,j,k);
                v_k0 = k ==  0 ? ll : surf(i,j,k-1);
                v_k1 = k == _K ? ll : surf(i,j,k+1);
            }
            else
            {
                v_i0 = i ==  0 ? 2.f*v-surf(i+1,j,k) : surf(i-1,j,k);
                v_i1 = i == _I ? 2.f*v-surf(i-1,j,k) : surf(i+1,j,k);
                v_j0 = j ==  0 ? 2.f*v-surf(i,j+1,k) : surf(i,j-1,k);
                v_j1 = j == _J ? 2.f*v-surf(i,j-1,k) : surf(i,j+1,k);
                v_k0 = k ==  0 ? 2.f*v-surf(i,j,k+1) : surf(i,j,k-1);
                v_k1 = k == _K ? 2.f*v-surf(i,j,k-1) : surf(i,j,k+1);
            }

            float dx[2] = { v-v_i0, v_i1-v };
            float dy[2] = { v-v_j0, v_j1-v };
            float dz[2] = { v-v_k0, v_k1-v };

            pTemp[ix] = v - dt * Max(sign, 0.f) * ( sqrt(Pow2(Max(dx[0],0.f))
                                                      + Pow2(Min(dx[1],0.f))
                                                      + Pow2(Max(dy[0],0.f))
                                                      + Pow2(Min(dy[1],0.f))
                                                      + Pow2(Max(dz[0],0.f))
                                                      + Pow2(Min(dz[1],0.f))) - 1.f )
                         - dt * Min(sign, 0.f) * ( sqrt(Pow2(Min(dx[0],0.f))
                                                      + Pow2(Max(dx[1],0.f))
                                                      + Pow2(Min(dy[0],0.f))
                                                      + Pow2(Max(dy[1],0.f))
                                                      + Pow2(Min(dz[0],0.f))
                                                      + Pow2(Max(dz[1],0.f))) - 1.f );

        };

        LaunchDeviceKernel( kernel, 0, surf.size() );
        SyncKernels();

        //float** ppSurf = surf.pPointer();
        //float** ppTemp = temp.pPointer();
        //Swap( *ppSurf, *ppTemp );
		Swap( surf, temp );
    }

}

VARE_NAMESPACE_END

