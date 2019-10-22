
#include <VARE.h>

VARE_NAMESPACE_BEGIN

FLIPWaterMethod::FLIPWaterMethod()
{
}

void FLIPWaterMethod::initialize( const Grid& grid )
{
    const size_t voxelNum = grid.numVoxels();
    if( voxelNum == 0 ) return;

    _collisions.initialize( grid );

    SparseFrame::remove( _frame );
    _frame = SparseFrame::create( grid, kUnified );

    _grid = _frame->grid();

    _sMarkField.initialize( _frame, 0    );
    _sMassField.initialize( _frame, 0.f  );
    _sSurfField.initialize( _frame, 10.f );
    _sPresField.initialize( _frame, 0.f  );
    _sVeloField.initialize( _frame, Vec3f(0.f) );
    _sCompField.initialize( _frame, Vec3f(0.f) );

    _scanField.initialize( _grid, kUnified );

    _isPrepared = false;

    _currentFrame = 0;

    // parameter for voxel-space
    _voxelExtForce = _grid.voxelVector( params.extForce );
    
    Vec3f height( 0.f, params.wallLevel, 0.f );
    height = _grid.voxelPoint( height );
    _voxelWallLevel = height.y;
}

void FLIPWaterMethod::prepareNextStep( Particles& pts )
{
    _frame->buildFromPoints( pts.position, params.enableWaterTank, _voxelWallLevel );
}

void FLIPWaterMethod::updateParticles( Particles& pts, const float dt )
{
    const VectorSparseField& compField = _sCompField;
    const VectorSparseField& veloField = _sVeloField;
    const ScalarSparseField& massField = _sMassField;
    const ScalarSparseField& surfField = _sSurfField;

    const Grid& grid = _grid;

    const float lx = (float)grid.nx()+0.5f;
    const float ly = (float)grid.ny()+0.5f;
    const float lz = (float)grid.nz()+0.5f;

    const size_t nx = grid.nx();
    const size_t ny = grid.ny();
    const size_t nz = grid.nz();

    const Params& prm = params;
    const VolumeSource& cols = _collisions;

    const bool enableTank = params.enableWaterTank;
    const float band = params.dampingBand / params.voxelSize;
    const float wall = _voxelWallLevel;
    const Vec3f& extForce = _voxelExtForce;

    Vec3f* p_pos = pts.position.pointer();
    Vec3f* p_vel = pts.velocity.pointer();
    float* p_adj = pts.adjacency.pointer();
    float* p_den = pts.density.pointer();

    char* rArr = _removedPts.pointer();

    auto kernel = VARE_DEVICE_KERNEL
    {
        Vec3f& pos = p_pos[ix];
        Vec3f& vel = p_vel[ix];

        p_adj[ix] = surfField.lerpMin( pos );

        /* update particle velocities */ 
        {
            Vec3f gridVel = compField.lerp( pos );
            Vec3f f = gridVel - veloField.lerp( pos );
                    
            vel += f;
            vel = vel*0.95f + gridVel*0.05f;
        }

        /* Damping particles near boundary */
        if( enableTank == true && pos.y <= wall )
        {
            float w(1.f);

            if( pos.x < band    ) w = Min( w, Clamp( (pos.x-0.f)/band, 0.f, 1.f ) );
            if( pos.x > lx-band ) w = Min( w, Clamp( (lx-pos.x )/band, 0.f, 1.f ) );

            if( pos.z < band    ) w = Min( w, Clamp( (pos.z-0.f)/band, 0.f, 1.f ) );
            if( pos.z > lz-band ) w = Min( w, Clamp( (lz-pos.z )/band, 0.f, 1.f ) );

            vel *= w;
        }

        if( cols.signedDistance( pos ) < 0.f )
        {
            pos += cols.velocity( pos ) * dt;
        }

        /* advect particles */
        {
            const int iter = 1 + Min( (int)(vel.length()*dt*1.5f), Max(prm.maxsubsteps,1) );
            const float sub_dt = dt / (float)iter;

            bool col(false);

            // substeps for advection.
            for( int x=0; x<iter; ++x )
            {
                pos += vel * sub_dt;

                // collision interaction.
                {
                    if( pos.y <= wall )
                    {
                        if( pos.x < 0.f ) { pos.x = 0.f; if( vel.x < 0.f ) vel.x = 0.f; }
                        if( pos.y < 0.f ) { pos.y = 0.f; if( vel.y < 0.f ) vel.y = 0.f; }
                        if( pos.z < 0.f ) { pos.z = 0.f; if( vel.z < 0.f ) vel.z = 0.f; }

                        if( pos.x > (float)nx ) { pos.x = (float)nx; if( vel.x > 0.f ) vel.x = 0.f; }
                        if( pos.y > (float)ny ) { pos.y = (float)ny; if( vel.y > 0.f ) vel.y = 0.f; }
                        if( pos.z > (float)nz ) { pos.z = (float)nz; if( vel.z > 0.f ) vel.z = 0.f; }
                    }

                    const float phi = cols.signedDistance( pos );
                    if( phi < 0.f )
                    {
                        Vec3f nor = cols.normal( pos ).normalized();
                        pos -= nor*phi;
                        
                        //nor = cols.normal( pos ).normalized();
                        //const float velNor = vel*nor;
                        //if( velNor < 0.f ) vel -= nor*velNor;
                        
                        col = true;
                    }
                }
            }

            if( col )
            {
                const float phi = cols.signedDistance( pos );
                if( phi > 3.f || phi < 0.f ) rArr[ix] = 1;
            }

            if( grid.inside( pos ) == false ) rArr[ix] = 1;
        }

        vel += extForce * dt;
    };

    LaunchDeviceKernel( kernel, 0, pts.position.size() );
    SyncKernels();
}

void FLIPWaterMethod::advanceOneFrame( Particles& pts )
{
    if( pts.position.size() == 0 || pts.velocity.size() == 0 ) return;

    watch.start( "Simulation Frame" );
    watch.start( "Advance One Step" );

    const float dt = params.dt;
    const float voxelSize = params.voxelSize;
    const Grid& grid = _grid;

    Vec3f* pos = pts.position.pointer();
    Vec3f* vel = pts.velocity.pointer();

    particlesToVoxel( pts );

    advanceOneStep( pts, dt );

    watch.stop();
    watch.start( "Post Processing" );

    postProcess( pts );

    particlesToWorld( pts );
    
    watch.stop();
    watch.stop();

    _currentFrame++;
}

void FLIPWaterMethod::advanceOneStep( Particles& pts, const float dt )
{
    if( _isPrepared == false ) prepareNextStep( pts );
    _isPrepared = false;

    const float voxelWallLevel = _voxelWallLevel;
    const float voxelSize = params.voxelSize;
    const float colVelScale = params.colVelScale;

    const IndexSparseField& markField = _sMarkField;
    const FloatSparseField& massField = _sMassField;
    const FloatSparseField& surfField = _sSurfField;
    const Vec3fSparseField& compField = _sCompField;
    const Vec3fSparseField& veloField = _sVeloField;

    const VolumeSource& cols = _collisions;

    size_t* pMark = markField.pointer();
    float*  pMass = massField.pointer();
    float*  pSurf = surfField.pointer();
    Vec3f*  pVelo = compField.pointer();
    Vec3f*  pPreV = veloField.pointer();

    _removedPts.initialize( pts.position.size(), kUnified );
    _removedPts.zeroize();

    Rasterization::ParticlesToVelField( _sMassField, _sVeloField, pts );
    Rasterization::PointsToCountField( _sSurfField, pts.position );

    const Noise& noise = curlNoise;
    const float frame = (float)_currentFrame;
    const float adjUnder = params.adjacencyUnder;
    const bool useCurl = params.enableCurl;

    auto mm_kernel = VARE_DEVICE_KERNEL
    {
        size_t i,j,k;
        massField.findIndices( ix, i,j,k );
        Vec3f cell = massField.cellCenter( i,j,k );
        const float phi = cols.signedDistance( cell );
        const Vec3f v = cols.velocity( cell );

        if( pMass[ix] > 0.f )
        {
            pMark[ix] = FluidProjector::kFluid;
        }
        else
        {
            if( phi < 0.f ) pMark[ix] = FluidProjector::kSolid;
            else pMark[ix] = FluidProjector::kAir;
        }

        if( phi < 0.f ) pPreV[ix] += v*colVelScale;

        /* add curl */
        if( useCurl && cell.y > voxelWallLevel-2.f && pMark[ix] != FluidProjector::kSolid )
        {
            const float m = pSurf[ix]-adjUnder;
            if( m < 0.f )
            {
                const float s = -m/adjUnder;

                const Vec3f world = massField.worldPoint( cell );
                const Vec3f curl = massField.voxelVector( noise.curl( world, frame*dt ) );
                
                pPreV[ix] = curl*s + pPreV[ix]*(1.f-s);
            }
        }

        pVelo[ix] = pPreV[ix];
    };
    
    LaunchDeviceKernel( mm_kernel, 0, _sMarkField.size() );
    SyncKernels();

    _projection.wallLevel = _voxelWallLevel;
    _projection.maxIteration = params.projIteration;

    _projection.buildLinearSystem( _sMarkField, _sCompField, _sPresField );
    const int projnum = _projection.solve( _sCompField, _sPresField, _sMarkField );


    updateParticles( pts, dt );

    // remove un-used particles.
    pts.position.remove( _removedPts );
    pts.velocity.remove( _removedPts );
    pts.vorticity.remove( _removedPts );
    pts.curvature.remove( _removedPts );
    pts.adjacency.remove( _removedPts );
}

void FLIPWaterMethod::postProcess( Particles& pts )
{
    prepareNextStep( pts );
    _isPrepared = true;

    const float voxelSize = params.voxelSize;
    const float dt = params.dt;

    const FloatSparseField& surfField = _sSurfField;
    const FloatSparseField& presField = _sPresField;
    const Vec3fSparseField& veloField = _sVeloField;
    const Vec3fSparseField& compField = _sCompField;
    const DenseField<char>  scanField = _scanField;

    const size_t nx = _grid.nx();
    const size_t ny = _grid.ny();
    const size_t nz = _grid.nz();

    Rasterization::ParticlesToSurfaceAndVelocity( _sSurfField, _sPresField, _sCompField, _sVeloField, pts, 1.5f, 0.5f );

    if( params.enableWaterTank == true )
    {
        _scanField.zeroize();

        std::vector<Vec3i> cells;
        cells.reserve( _grid.nx()*_grid.nz() );
        cells.push_back( Vec3i(nx-1, ny-1, nz-1) );

        const size_t wall = _voxelWallLevel;
        for( size_t k=0; k<nz; ++k )
        for( size_t i=0; i<nx; ++i )
        {
            if( surfField( i,wall,k ) > 0.f ) cells.push_back( Vec3i( i,wall,k ) );
        }

        while( cells.empty() == false )
        {
            const Vec3i& idx = cells[ cells.size()-1 ];
            cells.pop_back();

            _scanField(idx.i,idx.j,idx.k) = 1;
            if( idx.j > wall ) continue;
            
            if( idx.i-1 >= 0 && surfField(idx.i-1,idx.j,idx.k) > 0.f && _scanField(idx.i-1,idx.j,idx.k) == 0 ) cells.push_back( Vec3i(idx.i-1,idx.j,idx.k) );
            if( idx.i+1 < nx && surfField(idx.i+1,idx.j,idx.k) > 0.f && _scanField(idx.i+1,idx.j,idx.k) == 0 ) cells.push_back( Vec3i(idx.i+1,idx.j,idx.k) );
            if( idx.j-1 >= 0 && surfField(idx.i,idx.j-1,idx.k) > 0.f && _scanField(idx.i,idx.j-1,idx.k) == 0 ) cells.push_back( Vec3i(idx.i,idx.j-1,idx.k) );
            if( idx.j+1 < ny && surfField(idx.i,idx.j+1,idx.k) > 0.f && _scanField(idx.i,idx.j+1,idx.k) == 0 ) cells.push_back( Vec3i(idx.i,idx.j+1,idx.k) );
            if( idx.k-1 >= 0 && surfField(idx.i,idx.j,idx.k-1) > 0.f && _scanField(idx.i,idx.j,idx.k-1) == 0 ) cells.push_back( Vec3i(idx.i,idx.j,idx.k-1) );
            if( idx.k+1 < nz && surfField(idx.i,idx.j,idx.k+1) > 0.f && _scanField(idx.i,idx.j,idx.k+1) == 0 ) cells.push_back( Vec3i(idx.i,idx.j,idx.k+1) );
        }

        float* pSurf = surfField.pointer();
        auto mm_kernel = VARE_DEVICE_KERNEL
        {
            size_t i,j,k;
            surfField.findIndices( ix, i,j,k );
            if( j > wall ) return;

            if( pSurf[ix] > 0.f )
            {
                if( scanField( i,j,k ) == 0 ) pSurf[ix] = -5.f;
            }
        };

        LaunchDeviceKernel( mm_kernel, 0, surfField.size() );
        SyncKernels();
    }

    Rasterization::ReinitializeSurface( _sSurfField, _sPresField, params.redistIteration, params.enableWaterTank, _voxelWallLevel );

    FieldOP::curl( _sVeloField, _sCompField );

    Vec3f* pPose = pts.position.pointer();
    float* pVort = pts.vorticity.pointer();
    float* pCurv = pts.curvature.pointer();

    const float dis = Pow( 1.f-params.vortPresRate, dt );
    
    auto vort_kernel = VARE_DEVICE_KERNEL
    {
        const Vec3f& p = pPose[ix];
        float v = compField.worldVector( compField.lerp( p ) ).length();
        v = Max( 0.f, v-0.2f );

        pVort[ix] -= pVort[ix] * dis;
        pVort[ix] = Max( v, pVort[ix] );
    };

    LaunchDeviceKernel( vort_kernel, 0, pts.position.size() );
    SyncKernels();

    FieldOP::normalGradient( _sSurfField, _sCompField );
    FieldOP::curvature( _sCompField, _sPresField );

    auto curv_kernel = VARE_DEVICE_KERNEL
    {
        pCurv[ix] = presField.lerp( pPose[ix] );
    };

    LaunchDeviceKernel( curv_kernel, 0, pts.position.size() );
    SyncKernels(); 

    // clear
    _sPresField.zeroize();
    _sPresField.background = 0.f;
}

void FLIPWaterMethod::fillWaterTank( Particles& particles )
{
    if( _voxelWallLevel <= 0.f ) return;

    int nx = _grid.nx();
    int ny = _grid.ny();
    int nz = _grid.nz();

    int hh = Clamp( (int)_voxelWallLevel+1, 0, ny );

    for( int k=0; k<nz; ++k )
    for( int j=0; j<hh; ++j )
    for( int i=0; i<nx; ++i )
    {
        const Vec3f cen = _grid.cellCenter( i,j,k );
        const float band_width = _voxelWallLevel-cen.y;

        const float dx0 = Rand( i*45651+k*75561+j*65551, 0.0f, 1.0f );
        const float dy0 = Rand( i*51221+k*99331+j*11221, 0.0f, 1.0f );
        const float dz0 = Rand( i*81221+k*45331+j*66551, 0.0f, 1.0f );
        const float dx1 = Rand( i*53454+k*12234+j*23853, 0.0f, 1.0f );
        const float dy1 = Rand( i*76335+k*94762+j*93654, 0.0f, 1.0f );
        const float dz1 = Rand( i*98547+k*23443+j*53218, 0.0f, 1.0f );

        Vec3f p0 = cen - Vec3f(0.5f) + Vec3f(dx0, dy0, dz0);
        Vec3f p1 = cen - Vec3f(0.5f) + Vec3f(dx1, dy1, dz1);

        if( p0.y <= _voxelWallLevel )
        {
            Vec3f uvw( (float)p0.x/(float)(nx), (float)p0.y/(float)(ny), (float)p0.z/(float)(nz) );
            p0 = _grid.worldPoint(p0);
            newParticle( particles, p0, Vec3f(0.f) );
        }

        if( p1.y <= _voxelWallLevel )
        {
            Vec3f uvw( (float)p1.x/(float)(nx), (float)p1.y/(float)(ny), (float)p1.z/(float)(nz) );
            p1 = _grid.worldPoint(p1);
            newParticle( particles, p1, Vec3f(0.f) );
        }

        {
            Vec3f pp = cen;
            if( pp.y <= _voxelWallLevel )
            {
                Vec3f uvw( (float)pp.x/(float)(nx), (float)pp.y/(float)(ny), (float)pp.z/(float)(nz) );
                pp = _grid.worldPoint( pp );
                newParticle( particles, pp, Vec3f(0.f) );
            }
        }

        if( band_width > 0.f && band_width < 4.f )
        {
            const int N = params.ptsPerCell;
            for( int n=0; n<N; ++n )
            {
                const float dx = Rand( i*45651+k*75561+j*65551+n*44222, 0.0f, 1.0f );
                const float dy = Rand( i*51221+k*99331+j*11221+n*59374, 0.0f, 1.0f );
                const float dz = Rand( i*81221+k*45331+j*66551+n*12333, 0.0f, 1.0f );

                Vec3f pp = cen - Vec3f(0.5f) + Vec3f(dx, dy, dz);

                if( pp.y <= _voxelWallLevel )
                {
                    Vec3f uvw( (float)pp.x/(float)(nx), (float)pp.y/(float)(ny), (float)pp.z/(float)(nz) );
                    pp = _grid.worldPoint( pp );
                    newParticle( particles, pp, Vec3f(0.f) );
                }
            }
        }
    }
}

void
FLIPWaterMethod::particlesToWorld( Particles& pts )
{
    const Grid& grid = _grid;

    Vec3f* pos = pts.position.pointer();
    Vec3f* vel = pts.velocity.pointer();

    auto kernel_world = VARE_DEVICE_KERNEL
    {
        pos[ix] = grid.worldPoint( pos[ix] );
        vel[ix] = grid.worldVector( vel[ix] );
    };

    LaunchDeviceKernel( kernel_world, 0, pts.position.size() );
    SyncKernels();    
}

void
FLIPWaterMethod::particlesToVoxel( Particles& pts )
{
    const Grid& grid = _grid;

    Vec3f* pos = pts.position.pointer();
    Vec3f* vel = pts.velocity.pointer();
    
    auto kernel_voxel = VARE_DEVICE_KERNEL
    {
        pos[ix] = grid.voxelPoint( pos[ix] );
        vel[ix] = grid.voxelVector( vel[ix] );
    };

    LaunchDeviceKernel( kernel_voxel, 0, pts.position.size() );
    SyncKernels();    
}

VARE_NAMESPACE_END

