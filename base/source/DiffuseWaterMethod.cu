//-----------------------//
// DiffuseWaterMethod.cu //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2019.04.15                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

void
DiffuseWaterMethod::initialize( const Grid& g )
{
    _grid_frame = SparseFrame::create( g, kUnified );
    _grid = _grid_frame->_grid;

    _pts.position.initialize( 0, kUnified );
    _pts.velocity.initialize( 0, kUnified );
    _pts.lifespan.initialize( 0, kUnified );

    const size_t reserve_num = _grid.nx()*_grid.nz();

    _pts.position.reserve( reserve_num );
    _pts.velocity.reserve( reserve_num );
    _pts.lifespan.reserve( reserve_num );

    _fluidBody.initialize( g, true );
    _collisions.initialize( g );

    if( params.simType == 0 || params.simType == 1 )
    {
        _massField.initialize( _grid, kUnified );
        _veloField.initialize( _grid, kUnified );
        _diveField.initialize( _grid, kUnified );
        _presField.initialize( _grid, kUnified );
        _compField.initialize( _grid, kUnified );
        _areaField.initialize( _grid, kUnified );
        _markField.initialize( _grid, kUnified );

        _massField.zeroize();
        _veloField.zeroize();
        _areaField.zeroize();
        _markField.zeroize();
    }

    _currentFrame = 0;
    _seedNum = 1;

    // print out parameters

    std::cout<< "Type : " << params.simType << std::endl;
    std::cout<< "Voxel  Size : " << params.voxelSize << std::endl;
    std::cout<< "Gravity : " << params.gravity << std::endl;
    std::cout<< "Bouyancy : " << params.bouyancy << std::endl;
    std::cout<< "Drag Effect : " << params.dragEffect << std::endl;
    std::cout<< "Dissipation Water : " << params.disspWater << std::endl;
    std::cout<< "Dissipation Air : " << params.disspAir << std::endl;
    std::cout<< "Projection Iteration : " << params.iteration << std::endl;
    std::cout<< "Vortex Force : " << params.vortexForce << std::endl;
    std::cout<< "Potential Scale : " << params.potentialScale << std::endl;
    std::cout<< "Acceleration Scale : " << params.accelScale << std::endl;
    std::cout<< "Seed Scale : " << params.seedScale << std::endl;
    std::cout<< "Life Time : " << params.lifeTime << std::endl;
    std::cout<< "Life Variance : " << params.lifeVariance << std::endl;
    std::cout<< "Minimum Density : " << params.minDensity << std::endl;
}

void
DiffuseWaterMethod::generateSubParticles( const Vec3f& p, const Vec3f& v, const float voxel, const float dt, const size_t seed )
{
    float dist = v.length()*dt;

    int num = ((int)(dist/voxel)+1) * params.seedScale;
    
    const float r = dist*0.3f;
    Vec3f center = p;

    const float sub_dt = dt / (float)num;

    for( size_t x=1; x<=num; ++x )
    {
        const float rx = Rand( seed+x*332433 )-0.5f;
        const float ry = Rand( seed+x*756455 )-0.5f;
        const float rz = Rand( seed+x*285041 )-0.5f;
        const float rt = Rand( seed+x*128533 );

        const Vec3f POS = center + Vec3f( rx, ry, rz )*r;
        const Vec3f VEL = v + (POS-p)/dt * params.accelScale;

        _pts.position.append( POS );
        _pts.velocity.append( VEL );
        _pts.lifespan.append( params.lifeTime + params.lifeVariance*rt );

        center = POS + v*sub_dt;
    }
}

void
DiffuseWaterMethod::generateSplashParticles( const Particles& fluid_pts, const size_t num, const float voxelSize, const float dt )
{
    if( params.simType != 0 ) return;

    const Vec2f& cr = params.curvatureRange;

    for( size_t n=0; n<num; ++n )
    {
        const Vec3f& p = fluid_pts.position[n];
        const Vec3f& v = fluid_pts.velocity[n];

        float potential = (fluid_pts.curvature[n]-cr[0]) / ( cr[1]-cr[0] );
        if( fluid_pts.velocityNormal[n] < 0.3f ) potential = 0.f;

        potential = Clamp( potential, 0.f, 1.f ) * params.potentialScale;

        if( potential <= 0.f || Rand( _seedNum*n+459823 ) > potential ) continue;

        generateSubParticles( p, v, voxelSize, dt, n*_seedNum+234543 );
    }

    _seedNum += 1;
}

void
DiffuseWaterMethod::generateSprayParticles( const Particles& fluid_pts, const size_t num, const float voxelSize, const float dt )
{
    if( params.simType != 1 ) return;

    const Vec2f& cr = params.curvatureRange;

    for( size_t n=0; n<num; ++n )
    {
        const Vec3f& p = fluid_pts.position[n];
        const Vec3f& v = fluid_pts.velocity[n];

        float potential = (fluid_pts.curvature[n]-cr[0]) / ( cr[1]-cr[0] );
        if( fluid_pts.velocityNormal[n] < 0.3f ) potential = 0.f;

        potential = Clamp( potential, 0.f, 1.f ) * params.potentialScale;

        if( potential <= 0.f || Rand( _seedNum*n+538111 ) > potential ) continue;

        generateSubParticles( p, v, voxelSize, dt, n*_seedNum+376493 );
    }

    _seedNum += 1;
}

void
DiffuseWaterMethod::generateFoamParticles( const Particles& fluid_pts, const size_t num, const float voxelSize, const float dt )
{
    if( params.simType != 2 ) return;

    const Vec2f& cr = params.curvatureRange;

    for( size_t n=0; n<num; ++n )
    {
        const Vec3f& p = fluid_pts.position[n];
        const Vec3f& v = fluid_pts.velocity[n];

        float potential = (fluid_pts.curvature[n]-cr[0]) / ( cr[1]-cr[0] );
        if( Abs(fluid_pts.velocityNormal[n]) > 0.3f ) potential = 0.f;

        potential = Clamp( potential, 0.f, 1.f ) * params.potentialScale;

        if( potential <= 0.f || Rand( _seedNum*n+283341 ) > potential ) continue;

        generateSubParticles( p, v, voxelSize, dt, n*_seedNum+235349 );
    }

    _seedNum += 1;
}

void
DiffuseWaterMethod::generateBubbleParticles( const Particles& fluid_pts, const size_t num, const float voxelSize, const float dt )
{
    if( params.simType != 3 ) return;

    const Vec2f& cr = params.curvatureRange;    

    for( size_t n=0; n<num; ++n )
    {
        const Vec3f& p = fluid_pts.position[n];
        const Vec3f& v = fluid_pts.velocity[n];

        float potential = (fluid_pts.curvature[n]-cr[0]) / ( cr[1]-cr[0] );
        if( fluid_pts.velocityNormal[n] > -0.3f ) potential = 0.f;

        potential = Clamp( potential, 0.f, 1.f ) * params.potentialScale;

        if( potential <= 0.f || Rand( _seedNum*n+756611 ) > potential ) continue;

        generateSubParticles( p, v, voxelSize, dt, n*_seedNum+485343 );
    }

    _seedNum += 1;
}

void
DiffuseWaterMethod::seedParticles( PointArray& pose, Vec3fArray& velo )
{
    for( size_t n=0; n<pose.size(); ++n )
    {
        const float rt = Rand( _seedNum+n*128533 );
                
        _pts.position.append( pose[n] );
        _pts.velocity.append( velo[n] );
        _pts.lifespan.append( params.lifeTime + params.lifeVariance*rt );
    }

    _seedNum += pose.size();
}

void 
DiffuseWaterMethod::removeParticles( const float dt )
{
    if( _pts.position.size() == 0 ) return;

    CharArray& removedPts = _removedPts;
    Particles& pts = _pts;

    Grid& grid = _grid;

    float* lArr = pts.lifespan.pointer();
    Vec3f* pArr = pts.position.pointer();
    char*  rArr = removedPts.pointer();
    
    auto kernel_remove = [=] BORA_DEVICE ( const size_t n )
    {
        const Vec3f& p = pArr[n];

        lArr[n] -= dt;

        if( grid.inside( p, 3.f ) == false ) rArr[n] = 1;

        if( pts.lifespan[n] <= 0.f ) rArr[n] = 1;
    };

    LaunchCudaDevice( kernel_remove, 0, pts.position.size() );
    SyncCuda();

    // remove un-used particles.
    pts.position.remove( removedPts );
    pts.velocity.remove( removedPts );
    pts.lifespan.remove( removedPts );
}

void
DiffuseWaterMethod::advanceOneFrame( const int currframe, const float dt )
{
    Vec3f* pos = _pts.position.pointer();
    Vec3f* vel = _pts.velocity.pointer();

    Grid& grid = _grid;

    // world to voxel space;
    auto kernel_voxel = [=] BORA_DEVICE ( const size_t n )
    {
        pos[n] = grid.voxelPoint( pos[n] );
        vel[n] = grid.voxelVector( vel[n] );
    };

    LaunchCudaDevice( kernel_voxel, 0, _pts.position.size() );
    SyncCuda();


    advanceOneStep( dt );


    // voxel to world space;
    auto kernel_world = [=] BORA_DEVICE ( const size_t n )
    {
        pos[n] = grid.worldPoint( pos[n] );
        vel[n] = grid.worldVector( vel[n] );
    };

    LaunchCudaDevice( kernel_world, 0, _pts.position.size() );
    SyncCuda();

    _currentFrame++;
}

void
DiffuseWaterMethod::updateSplashParticles( const float dt )
{
    Particles& pts = _pts;
    const Params& prm = params;

    Vec3f voxelGravity = _grid.voxelVector( params.gravity );

    const size_t EI = _veloField.nx()-1;
    const size_t EJ = _veloField.ny()-1;
    const size_t EK = _veloField.nz()-1;

    const ScalarDenseField& massField = _massField;
    const VectorDenseField& veloField = _veloField;
    const ScalarDenseField& diveField = _diveField;
    const ScalarDenseField& presField = _presField;
    const DenseField<char>& markField = _markField;
    const VectorDenseField& compField = _compField;    

    float* pMass = _massField.pointer();
    char*  pMark = _markField.pointer();
    Vec3f* pVelo = _veloField.pointer();
    float* pDive = _diveField.pointer();
    float* pPres = _presField.pointer();
    Vec3f* pComp = _compField.pointer();

    const Grid& grid = _grid;
    const CollisionSource& body = _fluidBody;
    const CollisionSource& cols = _collisions;

    const Noise& noise = curlNoise;
    const float frame = (float)_currentFrame;
    const float adjUnder = params.adjacencyUnder;
    
    /* Rasterization step */
    Rasterization::ParticlesToVelField( _massField, _veloField, pts );
    Rasterization::PointsToCountField( _diveField, pts.position );

    auto kernel_mark = [=] BORA_DEVICE ( const size_t n )
    {
        size_t i,j,k;
        veloField.cellIndices( n, i, j, k );
        const Vec3f cell = veloField.cellCenter( i, j, k );

        pMark[n] = 0;

        if( pMass[n] > 1e-10f )
        {
            pMark[n] = 1;
        }
        else
        {
            if( body.signedDistance( cell ) <= 0.f )
            {
                pVelo[n] = body.velocity( cell );
                pMark[n] = 1;
            }

            if( cols.signedDistance( cell ) <= 0.f )
            {
                pVelo[n] = cols.velocity( cell );
                pMark[n] = 2;
            }
        }

        if( noise.scale > 0.f && pMark[n] != 2 )
        {
            const float m = pDive[n]-adjUnder;
            if( m < 0.f )
            {
                float s = -m/adjUnder;

                const Vec3f world = massField.worldPoint( cell );
                const Vec3f curl = massField.voxelVector( noise.curl( world, frame*dt ) );

                pVelo[n] = curl*s + pVelo[n]*(1.f-s);
            }
        }
    };

    LaunchCudaDevice( kernel_mark, 0, _massField.size() );
    SyncCuda();

    /* Divergence */
    auto kernel_dive = [=] BORA_DEVICE ( const size_t n )
    {
        size_t i,j,k;
        diveField.cellIndices( n, i, j, k );

        const Vec3f& v = pVelo[n];

        const float& v_i0 = i==0  ? v.x : markField(i-1,j,k)==0 ? v.x : veloField(i-1,j,k).x;
        const float& v_i1 = i==EI ? v.x : markField(i+1,j,k)==0 ? v.x : veloField(i+1,j,k).x;
        const float& v_j0 = j==0  ? v.y : markField(i,j-1,k)==0 ? v.y : veloField(i,j-1,k).y;
        const float& v_j1 = j==EJ ? v.y : markField(i,j+1,k)==0 ? v.y : veloField(i,j+1,k).y;
        const float& v_k0 = k==0  ? v.z : markField(i,j,k-1)==0 ? v.z : veloField(i,j,k-1).z;
        const float& v_k1 = k==EK ? v.z : markField(i,j,k+1)==0 ? v.z : veloField(i,j,k+1).z;

        pDive[n] = (v_i1-v_i0)*0.5f + (v_j1-v_j0)*0.5f + (v_k1-v_k0)*0.5f;
    };

    LaunchCudaDevice( kernel_dive, 0, _diveField.size() );
    SyncCuda();

    _presField.zeroize();
    _compField.zeroize();

    // jacobian projection step.
    auto kernel_proj = [=] BORA_DEVICE ( const size_t n )
    {
        size_t i,j,k;
        presField.cellIndices( n, i, j, k );

        const float& p = presField[n];
        
        const float p_i0 = i==0  ? -p : markField(i-1,j,k)==0 ? -p : markField(i-1,j,k)==2 ? p : presField(i-1,j,k);
        const float p_i1 = i==EI ? -p : markField(i+1,j,k)==0 ? -p : markField(i+1,j,k)==2 ? p : presField(i+1,j,k);
        const float p_j0 = j==0  ? -p : markField(i,j-1,k)==0 ? -p : markField(i,j-1,k)==2 ? p : presField(i,j-1,k);
        const float p_j1 = j==EJ ? -p : markField(i,j+1,k)==0 ? -p : markField(i,j+1,k)==2 ? p : presField(i,j+1,k);
        const float p_k0 = k==0  ? -p : markField(i,j,k-1)==0 ? -p : markField(i,j,k-1)==2 ? p : presField(i,j,k-1);
        const float p_k1 = k==EK ? -p : markField(i,j,k+1)==0 ? -p : markField(i,j,k+1)==2 ? p : presField(i,j,k+1);

        pComp[n].x = ( -diveField[n] + p_i0 + p_i1 + p_j0 + p_j1 + p_k0 + p_k1 )/6.f;
    };

    auto kernel_swap = [=] BORA_DEVICE ( const size_t n )
    {
        pPres[n] = pComp[n].x;
    };

    for( size_t n=0; n<params.iteration; ++n )
    {
        LaunchCudaDevice( kernel_proj, 0, _presField.size() );
        SyncCuda();

        LaunchCudaDevice( kernel_swap, 0, _presField.size() );
        SyncCuda();
    }

    auto kernel_proj_grad = [=] BORA_DEVICE ( const size_t n )
    {
        size_t i,j,k;
        presField.cellIndices( n, i, j, k );

        const float& p = presField[n];

        const float p_i0 = i==0  ? -p : markField(i-1,j,k)==0 ? -p : markField(i-1,j,k)==2 ? p : presField(i-1,j,k);
        const float p_i1 = i==EI ? -p : markField(i+1,j,k)==0 ? -p : markField(i+1,j,k)==2 ? p : presField(i+1,j,k);
        const float p_j0 = j==0  ? -p : markField(i,j-1,k)==0 ? -p : markField(i,j-1,k)==2 ? p : presField(i,j-1,k);
        const float p_j1 = j==EJ ? -p : markField(i,j+1,k)==0 ? -p : markField(i,j+1,k)==2 ? p : presField(i,j+1,k);
        const float p_k0 = k==0  ? -p : markField(i,j,k-1)==0 ? -p : markField(i,j,k-1)==2 ? p : presField(i,j,k-1);
        const float p_k1 = k==EK ? -p : markField(i,j,k+1)==0 ? -p : markField(i,j,k+1)==2 ? p : presField(i,j,k+1);

        Vec3f grad( (p_i1-p_i0)*0.5f, (p_j1-p_j0)*0.5f, (p_k1-p_k0)*0.5f );
        pComp[n] = pVelo[n] - grad;
    };

    LaunchCudaDevice( kernel_proj_grad, 0, _presField.size() );
    SyncCuda();

    /* Particle update step */
    Vec3f* pt_pos = pts.position.pointer();
    Vec3f* pt_vel = pts.velocity.pointer();
    char*  prmv   = _removedPts.pointer();

    auto kernel_pts_update = [=] BORA_DEVICE( const size_t n )
    {
        Vec3f& p = pt_pos[n];
        Vec3f& v = pt_vel[n];

        const Vec3f gridvel = compField.lerp( p );
        const Vec3f force = gridvel - veloField.lerp( p );

        v += force;
        v = v*0.95f + gridvel*0.05f;

        if( cols.signedDistance(p) <= 0.f )
        {
            p += cols.velocity(p) * dt;
        }

        const float bodyPhi = body.signedDistance(p); 
        const Vec3f bodyNorm = body.normal(p).normalized();
        const float normVeloMag = bodyNorm * v;

        if( bodyPhi <= 0.f && normVeloMag < 0.f )
        {
            const Vec3f norVelo = bodyNorm * normVeloMag;
            const Vec3f tanVelo = v - norVelo;

            v = tanVelo - norVelo*0.3f;
        }

        if( bodyPhi > 0.f )
        {
            v += voxelGravity * dt;
        }

        if( bodyNorm * v.normalized() < 0.02f && bodyPhi < 0.f )
        {
            prmv[n] = 1;
        }

        p += v * dt;

        if( bodyPhi <= -3.f ) prmv[n] = 1;

        if( cols.signedDistance(p) <= 0.f ) prmv[n] = 1;
    };

    LaunchCudaDevice( kernel_pts_update, 0, _pts.position.size() );
    SyncCuda();
}

void
DiffuseWaterMethod::updateSprayParticles( const float dt )
{
    Particles& pts = _pts;
    const Params& prm = params;

    Vec3f voxelGravity = _grid.voxelVector( params.gravity );

    const ScalarDenseField& massField = _massField;
    const ScalarDenseField& diveField = _diveField;
    const ScalarDenseField& presField = _presField;
    const ScalarDenseField& areaField = _areaField;
    const VectorDenseField& veloField = _veloField;
    const VectorDenseField& compField = _compField;
    const DenseField<char>& markField = _markField;

    float* pMass = _massField.pointer();
    float* pDive = _diveField.pointer();
    float* pPres = _presField.pointer();
    float* pArea = _areaField.pointer();
    Vec3f* pVelo = _veloField.pointer();
    Vec3f* pComp = _compField.pointer();
    char*  pMark = _markField.pointer();

    const size_t EI = presField.nx()-1;
    const size_t EJ = presField.ny()-1;
    const size_t EK = presField.nz()-1;

    const Grid& grid = _grid;
    const CollisionSource& body = _fluidBody;
    const CollisionSource& cols = _collisions;
    
    /* Rasterization step */
    Rasterization::ParticlesToVelField( _diveField, _compField, pts );

    const Noise& noise = curlNoise;
    const float frame = (float)_currentFrame;

    auto kernel_rest = [=] BORA_DEVICE ( const size_t n )
    {
        pMark[n] = 0;

        size_t i,j,k;
        veloField.cellIndices( n, i, j, k );
        Vec3f cell = veloField.cellCenter( i,j,k );

        const float& m = pDive[n];
        if( m > 1e-10f )
        {
            pMass[n] = Max( pMass[n], (float)m-prm.minDensity );
            pArea[n] = 1.f;
            pVelo[n] = pComp[n];
        }
        else
        {
            if( cols.signedDistance( cell ) <= 0.f )
            {
                pVelo[n] = cols.velocity( cell );
                pMark[n] = 1;
            }
        }
    };

    LaunchCudaDevice( kernel_rest, 0, _veloField.size() );
    SyncCuda();

    {
        FieldOP::curl( _veloField, _compField );

        auto kernel_vortex = [=] BORA_DEVICE ( const size_t n )
        {
            size_t i,j,k;
            diveField.cellIndices( n, i,j,k );

            if( i >= EI-1 || j >= EJ-1 || k >= EK-1 || i <= 1 || j <= 1 || k <= 1 ) return;

            const float v = compField[n].length();
            const float v_i0 = i ==  0 ? v*2.f-compField(i+1,j,k).length() : compField(i-1,j,k).length();
            const float v_i1 = i == EI ? v*2.f-compField(i-1,j,k).length() : compField(i+1,j,k).length();
            const float v_j0 = j ==  0 ? v*2.f-compField(i,j+1,k).length() : compField(i,j-1,k).length();
            const float v_j1 = j == EJ ? v*2.f-compField(i,j-1,k).length() : compField(i,j+1,k).length();
            const float v_k0 = k ==  0 ? v*2.f-compField(i,j,k+1).length() : compField(i,j,k-1).length();
            const float v_k1 = k == EK ? v*2.f-compField(i,j,k-1).length() : compField(i,j,k+1).length();

            Vec3f nor( (v_i1-v_i0)*0.5f, (v_j1-v_j0)*0.5f, (v_k1-v_k0)*0.5f );

            if( nor.length() > 1e-10f )
            {
                const float alpha = Max( 1.f-areaField[n], 0.f );
                pVelo[n] += (nor.normalized()^compField[n]) * prm.vortexForce * alpha;
            }

        };

        LaunchCudaDevice( kernel_vortex, 0, _veloField.size() );
        SyncCuda();
    }

    /* Projection Step */
    FieldOP::divergence( _veloField, _diveField );

    _presField.zeroize();
    _compField.zeroize();

    // jacobian projection step.
    auto kernel_proj = [=] BORA_DEVICE ( const size_t n )
    {
        size_t i,j,k;
        presField.cellIndices( n, i, j, k );

        const float& p = presField[n];
        
        const float p_i0 = i ==  0 ? -p : markField(i-1,j,k)>0 ? p : presField(i-1,j,k);
        const float p_i1 = i == EI ? -p : markField(i+1,j,k)>0 ? p : presField(i+1,j,k);
        const float p_j0 = j ==  0 ? -p : markField(i,j-1,k)>0 ? p : presField(i,j-1,k);
        const float p_j1 = j == EJ ? -p : markField(i,j+1,k)>0 ? p : presField(i,j+1,k);
        const float p_k0 = k ==  0 ? -p : markField(i,j,k-1)>0 ? p : presField(i,j,k-1);
        const float p_k1 = k == EK ? -p : markField(i,j,k+1)>0 ? p : presField(i,j,k+1);

        pComp[n].x = ( -diveField[n] + p_i0 + p_i1 + p_j0 + p_j1 + p_k0 + p_k1 )/6.f;
    };

    auto kernel_swap = [=] BORA_DEVICE ( const size_t n )
    {
        pPres[n] = pComp[n].x;
    };

    for( size_t n=0; n<params.iteration; ++n )
    {
        LaunchCudaDevice( kernel_proj, 0, _presField.size() );
        SyncCuda();

        LaunchCudaDevice( kernel_swap, 0, _presField.size() );
        SyncCuda();
    }

    auto kernel_proj_grad = [=] BORA_DEVICE ( const size_t n )
    {
        size_t i,j,k;
        presField.cellIndices( n, i, j, k );

        const float& p = presField[n];

        const float p_i0 = i ==  0 ? -p : markField(i-1,j,k)>0 ? p : presField(i-1,j,k);
        const float p_i1 = i == EI ? -p : markField(i+1,j,k)>0 ? p : presField(i+1,j,k);
        const float p_j0 = j ==  0 ? -p : markField(i,j-1,k)>0 ? p : presField(i,j-1,k);
        const float p_j1 = j == EJ ? -p : markField(i,j+1,k)>0 ? p : presField(i,j+1,k);
        const float p_k0 = k ==  0 ? -p : markField(i,j,k-1)>0 ? p : presField(i,j,k-1);
        const float p_k1 = k == EK ? -p : markField(i,j,k+1)>0 ? p : presField(i,j,k+1);

        Vec3f grad( (p_i1-p_i0)*0.5f, (p_j1-p_j0)*0.5f, (p_k1-p_k0)*0.5f );
        pComp[n] = pVelo[n] - grad;
    };

    LaunchCudaDevice( kernel_proj_grad, 0, _presField.size() );
    SyncCuda();

    Vec3f* pt_pos  = pts.position.pointer();
    Vec3f* pt_vel  = pts.velocity.pointer();
    char*  prmv    = _removedPts.pointer();

    /* Particle update step */
    auto kernel_pts_update = [=] BORA_DEVICE ( const size_t n )
    {        
        Vec3f& p = pt_pos[n];
        Vec3f& v = pt_vel[n];

        const Vec3f gridvel = compField.lerp( p );
        const Vec3f force = gridvel - veloField.lerp( p );

        v += force;
        v = v*0.95f + gridvel*0.05f;

        if( cols.signedDistance(p) <= 0.f )
        {
            p += cols.velocity(p) * dt;
        }

        const float bodyPhi = body.signedDistance(p); 
        const Vec3f bodyNorm = body.normal(p).normalized();
        const float normVeloMag = bodyNorm * v;

        if( bodyPhi <= 0.f && normVeloMag < 0.f )
        {
            const Vec3f norVelo = bodyNorm * normVeloMag;
            const Vec3f tanVelo = v - norVelo;

            v = tanVelo - norVelo*0.1f;
        }

        if( bodyPhi > 0.f )
        {
            v += voxelGravity * dt;
        }

        if( bodyNorm * v.normalized() < 0.02f && bodyPhi < 0.f )
        {
            prmv[n] = 1;
        }

        p += v * dt;

        if( bodyPhi <= -3.f ) prmv[n] = 1;

        if( cols.signedDistance(p) <= 0.f ) prmv[n] = 1;
    };

    LaunchCudaDevice( kernel_pts_update, 0, _pts.position.size() );
    SyncCuda();

    {
        // Semi-Lagrangian Density Advection.
        auto kernel_semi_adv = [=] BORA_DEVICE ( const size_t n )
        {
            size_t i,j,k;
            massField.cellIndices( n, i, j, k );

            Vec3f p = massField.cellCenter( i,j,k );
            Vec3f v = compField.lerp( p );

            const float dist = v.length()*dt;

            const int iter = Min( (int)(dist / 1.f)+1, 10 );
            const float adv_dt = dt/(float)iter;

            for( int x=0; x<iter; ++x )
            {
                p -= v * adv_dt;
                v = compField.lerp( p );
            }

            pDive[n] = massField.lerp( p );
            pPres[n] = areaField.lerp( p );
            pVelo[n] = compField.lerp( p );

            const float& alpha = pArea[n];

            const float d_WaterRate = Pow( prm.disspWater, dt );
            const float d_AirRate = Pow( prm.disspAir, dt ) * Max( 1.f-alpha, 0.f );

            // dissipation under water.
            const float& phi = body.signedDistance( p );
            if( phi <= 0.f )
            {
                pDive[n] -= pDive[n] * d_WaterRate;
            }
            else
            {
                pDive[n] -= pDive[n] * d_AirRate;
            }

            // external force
            pVelo[n] += voxelGravity * alpha * dt;

            // curl noise
            if( pDive[n] <= 0.f )
            {
                Vec3f world = compField.worldPoint( p );
                const Vec3f curl = compField.voxelVector( noise.curl( world, frame*dt ) );
                pVelo[n] = curl;
            }
        };

        LaunchCudaDevice( kernel_semi_adv, 0, _massField.size() );
        SyncCuda();

        // Swap
        float** ppMass = _massField.pPointer();
        float** ppDive = _diveField.pPointer();
        Swap( *ppMass, *ppDive );

        pMass = _massField.pointer();
        pDive = _diveField.pointer();

        float** ppAres = _areaField.pPointer();
        float** ppPres = _presField.pPointer();
        Swap( *ppAres, *ppPres );
        
        pArea = _areaField.pointer();
        pPres = _presField.pointer();
    }
}

void
DiffuseWaterMethod::updateBubbleParticles( const float dt )
{
    const Params& prm = params;
    
    Vec3f* pos = _pts.position.pointer();
    Vec3f* vel = _pts.velocity.pointer();
    char*  typ = _pts.type.pointer();
    char*  rmv = _removedPts.pointer();

    CollisionSource& body = _fluidBody;

    Vec3f voxelGravity = _grid.voxelVector( params.gravity );

    auto kernel_bubble = [=] BORA_DEVICE ( const size_t n )
    {
        Vec3f g_vel = body.velocity( pos[n] );

        vel[n] += (g_vel-vel[n])*prm.dragEffect - voxelGravity*dt*prm.bouyancy;

        pos[n] += vel[n] * dt;

        const float phi = body.signedDistance( pos[n] );
        if( phi >= -0.5f ) rmv[n] = 1;
    };

    LaunchCudaDevice( kernel_bubble, 0, _pts.position.size() );
    SyncCuda();
}

void 
DiffuseWaterMethod::updateFoamParticles( const float dt )
{    
    Vec3f* pos = _pts.position.pointer();
    Vec3f* vel = _pts.velocity.pointer();
    char*  rmv = _removedPts.pointer();

    const CollisionSource& cols = _collisions;
    const CollisionSource& body = _fluidBody;

    Vec3f voxelGravity = _grid.voxelVector( params.gravity );

    auto kernel_foam = [=] BORA_DEVICE ( const size_t n )
    {
        Vec3f& p = pos[n];
        Vec3f& v = vel[n];

        const int subs = (int)v.length()+1;
        const float sub_dt = dt / (float)subs;

        for( int x=0; x<subs; ++x )
        {
            p += v * sub_dt;

            const float phi = body.signedDistance( p )-0.05f;

            if( phi > 0.f )
            {
                v += voxelGravity * sub_dt;
            }
            else
            {
                const Vec3f nor = body.normal( p ).normalized();
                p -= nor * phi;

                v = body.velocity( p );
            }

            if( Abs( phi ) > 2.f )
            {
                rmv[n] = 1;
                break;
            }
        }

        if( cols.signedDistance( p ) < 0.f ) rmv[n] = 1;
    };

    LaunchCudaDevice( kernel_foam, 0, _pts.position.size() );
    SyncCuda();

}

void
DiffuseWaterMethod::advanceOneStep( const float sub_dt )
{
    if( _pts.position.size() == 0 && params.simType == 1 ) return;

    _removedPts.initialize( _pts.position.size(), kUnified );
    _removedPts.zeroize();

    if( params.simType == 0 ) updateSplashParticles( sub_dt );
    if( params.simType == 1 ) updateSprayParticles( sub_dt );
    if( params.simType == 2 ) updateFoamParticles( sub_dt );
    if( params.simType == 3 ) updateBubbleParticles( sub_dt );

    removeParticles( sub_dt );
}

void
DiffuseWaterMethod::updateFluidParticles( Particles& pts )
{
    pts.curvature.initialize( pts.position.size(), kUnified );
    pts.velocityNormal.initialize( pts.position.size(), kUnified );

    Grid& grid = _grid;

    const CollisionSource& body = _fluidBody;
    const float voxelSize = params.voxelSize;

    Vec3f* pPos = pts.position.pointer();
    float* pVN = pts.velocityNormal.pointer();
    float* pCur = pts.curvature.pointer();

    auto kernel_norm = [=] BORA_DEVICE ( const size_t n )
    {
        Vec3f p = grid.voxelPoint( pts.position[n] );

        const Vec3f v = grid.voxelVector( pts.velocity[n] );
        const Vec3f N = body.normal( p ).normalized();
        const float phi = body.signedDistance( p );

        if( Abs(phi) < 5.f )
        {
            p = p - N * phi;
            pPos[n] = grid.worldPoint( p );
        }

        pCur[n] = body.curvature( p );
        pVN[n] = body.normal(p).normalized()*v.normalized();
    };

    LaunchCudaDevice( kernel_norm, 0, pts.position.size() );
    SyncCuda();
}

BORA_NAMESPACE_END

