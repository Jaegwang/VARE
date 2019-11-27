//---------------------//
// FluidProjection.cpp //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.11.07                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

FluidProjection::FluidProjection()
{
    _fluidCellBuffer.initialize( 0, MemorySpace::kUnified );
}

void
FluidProjection::buildLinearSystem( DenseField<size_t>& typField, const VectorDenseField& velField, const ScalarDenseField& pressField )
{
    const size_t N = typField.numVoxels();

    _fluidCellBuffer.clear();    
    _fluidCellBuffer.reserve( N/3+1 );

    const Grid& grid = typField.getGrid();

    size_t count = 0;
    for( int n=0; n<N; ++n )
    {
        if( typField[n] >= kFluid )
        {
            typField[n] = count+kFluid;
            _fluidCellBuffer.append( n );
            count++;
        }
    }

    _A.initialize( count, 7 );
    _x.initialize( count, MemorySpace::kUnified );
    _b.initialize( count, MemorySpace::kUnified );
    
    size_t I = typField.nx()-1;
    size_t J = typField.ny()-1;
    size_t K = typField.nz()-1;

    for( int n=0; n<count; ++n )
    {
        size_t i,j,k;
        typField.cellIndices( n, i,j,k );

        size_t wall = (size_t)kSolid;

        size_t t_cc = typField( i,j,k );
        size_t t_i0 = i == 0 ? wall : typField( i-1,j,k );
        size_t t_i1 = i == I ? wall : typField( i+1,j,k );
        size_t t_j0 = j == 0 ? wall : typField( i,j-1,k );
        size_t t_j1 = j == J ? wall : typField( i,j+1,k );
        size_t t_k0 = k == 0 ? wall : typField( i,j,k-1 );
        size_t t_k1 = k == K ? wall : typField( i,j,k+1 );

        Vec3f v = velField( i,j,k );

        float v_i0 = t_i0 == kAir ? v.x : t_i0 == kSolid ? Max(v.x,-v.x) : velField( i-1,j,k ).x;
        float v_i1 = t_i1 == kAir ? v.x : t_i1 == kSolid ? Min(v.x,-v.x) : velField( i+1,j,k ).x;
        float v_j0 = t_j0 == kAir ? v.y : t_j0 == kSolid ? Max(v.y,-v.y) : velField( i,j-1,k ).y;
        float v_j1 = t_j1 == kAir ? v.y : t_j1 == kSolid ? Min(v.y,-v.y) : velField( i,j+1,k ).y;
        float v_k0 = t_k0 == kAir ? v.z : t_k0 == kSolid ? Max(v.z,-v.z) : velField( i,j,k-1 ).z;
        float v_k1 = t_k1 == kAir ? v.z : t_k1 == kSolid ? Min(v.z,-v.z) : velField( i,j,k+1 ).z;

        float p(6.f),p_i0(-1.f),p_i1(-1.f),p_j0(-1.f),p_j1(-1.f),p_k0(-1.f),p_k1(-1.f);
        const float d = -( (v_i1-v_i0)/_DX*0.5f + (v_j1-v_j0)/_DX*0.5f + (v_k1-v_k0)/_DX*0.5f ) * _DX*_DX;

        if(t_i0 == kAir) { p+=1.f; p_i0=0.f; } else if(t_i0 == kSolid) { p-=1.f; p_i0=0.f; }
        if(t_i1 == kAir) { p+=1.f; p_i1=0.f; } else if(t_i1 == kSolid) { p-=1.f; p_i1=0.f; }

        if(t_j0 == kAir) { p+=1.f; p_j0=0.f; } else if(t_j0 == kSolid) { p-=1.f; p_j0=0.f; }
        if(t_j1 == kAir) { p+=1.f; p_j1=0.f; } else if(t_j1 == kSolid) { p-=1.f; p_j1=0.f; }

        if(t_k0 == kAir) { p+=1.f; p_k0=0.f; } else if(t_k0 == kSolid) { p-=1.f; p_k0=0.f; }
        if(t_k1 == kAir) { p+=1.f; p_k1=0.f; } else if(t_k1 == kSolid) { p-=1.f; p_k1=0.f; }

        float* vals = _A.valuesOnRow( n );
        size_t* inds = _A.indicesOnRow( n );

        vals[0] = p_i0;
        vals[1] = p_i1;
        vals[2] = p_j0;
        vals[3] = p_j1;
        vals[4] = p_k0;
        vals[5] = p_k1;
        vals[6] = p;

        inds[0] = t_i0 >= kFluid ? t_i0-kFluid : NULL_MAX;
        inds[1] = t_i1 >= kFluid ? t_i1-kFluid : NULL_MAX;
        inds[2] = t_j0 >= kFluid ? t_j0-kFluid : NULL_MAX;
        inds[3] = t_j1 >= kFluid ? t_j1-kFluid : NULL_MAX;
        inds[4] = t_k0 >= kFluid ? t_k0-kFluid : NULL_MAX;
        inds[5] = t_k1 >= kFluid ? t_k1-kFluid : NULL_MAX;
        inds[6] = t_cc-kFluid;

        _x[n] = pressField(i,j,k);
        _b[n] = d;
    }

    //_A.printFile( "matrixA.txt" );
    //_b.printFile( "vectorb.txt" );
}

int
FluidProjection::solve( VectorDenseField& velField, ScalarDenseField& _pressureField, DenseField<size_t>& typField )
{
    /* Solve linear system */
    int iternum = LinearSystemSolver::cg( _A, _x, _b, maxIteration );
    //_x.printFile( "vectorx.txt" );

    /* Build pressure field */
    _pressureField.setValueAll( 0.f );
    for( size_t n=0; n<_fluidCellBuffer.size(); ++n )
    {
        size_t idx = _fluidCellBuffer[n];
        size_t i,j,k;

        typField.cellIndices( idx, i,j,k );

        size_t p = typField[idx]-kFluid;
        _pressureField[idx] = _x[ p ];
    }

    /* Gradient pressure field */
    size_t I = typField.nx()-1;
    size_t J = typField.ny()-1;
    size_t K = typField.nz()-1;

    size_t wall = (size_t)kSolid;

    for( int n=0; n<_pressureField.numVoxels(); ++n )
    {
        size_t i,j,k;
        _pressureField.cellIndices( n, i,j,k );

        float p = _pressureField[n];
        
        size_t t_i0 = i == 0 ? wall : typField( i-1,j,k );
        size_t t_i1 = i == I ? wall : typField( i+1,j,k );
        size_t t_j0 = j == 0 ? wall : typField( i,j-1,k );
        size_t t_j1 = j == J ? wall : typField( i,j+1,k );
        size_t t_k0 = k == 0 ? wall : typField( i,j,k-1 );
        size_t t_k1 = k == K ? wall : typField( i,j,k+1 );    

        const float p_i0 = t_i0 == kSolid ? p : t_i0 == kAir ? -p : _pressureField( i-1,j,k );
        const float p_i1 = t_i1 == kSolid ? p : t_i1 == kAir ? -p : _pressureField( i+1,j,k );
        const float p_j0 = t_j0 == kSolid ? p : t_j0 == kAir ? -p : _pressureField( i,j-1,k );
        const float p_j1 = t_j1 == kSolid ? p : t_j1 == kAir ? -p : _pressureField( i,j+1,k );
        const float p_k0 = t_k0 == kSolid ? p : t_k0 == kAir ? -p : _pressureField( i,j,k-1 );
        const float p_k1 = t_k1 == kSolid ? p : t_k1 == kAir ? -p : _pressureField( i,j,k+1 );

        Vec3f grad( (p_i1-p_i0)/_DX*0.5f, (p_j1-p_j0)/_DY*0.5f, (p_k1-p_k0)/_DZ*0.5f );
        
        velField[n] -= grad;
    }

    return iternum;
}

void
FluidProjection::buildLinearSystem( IdxSparseField& typField, const Vec3fSparseField& veloField, const FloatSparseField& pressField )
{
    const size_t N = typField.size();

    IndexArray& fluidCellBuffer = _fluidCellBuffer;
    SparseMatrix<float>& A = _A;
    FloatArray& M = _M;

    _fluidCellBuffer.clear();
    _fluidCellBuffer.reserve( N );

    size_t count(0);
    for( int n=0; n<N; ++n )
    {
        if( typField[n] >= kFluid )
        {
            typField[n] = count+kFluid;
            _fluidCellBuffer.append( n );
            count++;
        }
    }

    const size_t EI = typField.nx()-1;
    const size_t EJ = typField.ny()-1;
    const size_t EK = typField.nz()-1;

    _A.initialize( count, 7 );
    _M.initialize( count, MemorySpace::kUnified );

    _x.initialize( count, MemorySpace::kUnified );
    _b.initialize( count, MemorySpace::kUnified );

    float* px = _x.pointer();
    float* pb = _b.pointer();

    const float wallLv = wallLevel;

    auto kernel = [=] BORA_DEVICE ( const size_t n )
    {
        size_t i,j,k;
        typField.findIndices( fluidCellBuffer[n], i,j,k );

        size_t wall = (float)j+0.5f <= wallLv+1.f ? (size_t)kWall : (size_t)kAir;

        const size_t t_cc = typField( i,j,k );
        const size_t t_i0 = i ==  0 ? wall : typField( i-1,j,k );
        const size_t t_i1 = i == EI ? wall : typField( i+1,j,k );
        const size_t t_j0 = j ==  0 ? wall : typField( i,j-1,k );
        const size_t t_j1 = j == EJ ? wall : typField( i,j+1,k );
        const size_t t_k0 = k ==  0 ? wall : typField( i,j,k-1 );
        const size_t t_k1 = k == EK ? wall : typField( i,j,k+1 );

        const Vec3f v = veloField( i,j,k );

        float i0_j_k = i ==  0 ? 2.f*v.x-veloField( i+1,j,k ).x : veloField( i-1,j,k ).x;
        float i1_j_k = i == EI ? 2.f*v.x-veloField( i-1,j,k ).x : veloField( i+1,j,k ).x;
        float i_j0_k = j ==  0 ? 2.f*v.y-veloField( i,j+1,k ).y : veloField( i,j-1,k ).y;
        float i_j1_k = j == EJ ? 2.f*v.y-veloField( i,j-1,k ).y : veloField( i,j+1,k ).y;
        float i_j_k0 = k ==  0 ? 2.f*v.z-veloField( i,j,k+1 ).z : veloField( i,j,k-1 ).z;
        float i_j_k1 = k == EK ? 2.f*v.z-veloField( i,j,k-1 ).z : veloField( i,j,k+1 ).z;

        float v_i0 = t_i0 == kAir ? 2.f*v.x-i1_j_k : t_i0 == kWall ?  Abs(v.x) : i0_j_k;
        float v_i1 = t_i1 == kAir ? 2.f*v.x-i0_j_k : t_i1 == kWall ? -Abs(v.x) : i1_j_k;
        float v_j0 = t_j0 == kAir ? 2.f*v.y-i_j1_k : t_j0 == kWall ?  Abs(v.y) : i_j0_k;
        float v_j1 = t_j1 == kAir ? 2.f*v.y-i_j0_k : t_j1 == kWall ? -Abs(v.y) : i_j1_k;
        float v_k0 = t_k0 == kAir ? 2.f*v.z-i_j_k1 : t_k0 == kWall ?  Abs(v.z) : i_j_k0;
        float v_k1 = t_k1 == kAir ? 2.f*v.z-i_j_k0 : t_k1 == kWall ? -Abs(v.z) : i_j_k1;

        float p(6.f),p_i0(0.f),p_i1(0.f),p_j0(0.f),p_j1(0.f),p_k0(0.f),p_k1(0.f);
        float d = -( (v_i1-v_i0)*0.5f + (v_j1-v_j0)*0.5f + (v_k1-v_k0)*0.5f );

        if(t_i0 == kAir) { p+=1.f; } else if(t_i0 == kSolid || t_i0 == kWall) { p-=1.f; } else if(t_i0 >= kFluid) { p_i0-=1.f; }
        if(t_i1 == kAir) { p+=1.f; } else if(t_i1 == kSolid || t_i1 == kWall) { p-=1.f; } else if(t_i1 >= kFluid) { p_i1-=1.f; }

        if(t_j0 == kAir) { p+=1.f; } else if(t_j0 == kSolid || t_j0 == kWall) { p-=1.f; } else if(t_j0 >= kFluid) { p_j0-=1.f; }
        if(t_j1 == kAir) { p+=1.f; } else if(t_j1 == kSolid || t_j1 == kWall) { p-=1.f; } else if(t_j1 >= kFluid) { p_j1-=1.f; }

        if(t_k0 == kAir) { p+=1.f; } else if(t_k0 == kSolid || t_k0 == kWall) { p-=1.f; } else if(t_k0 >= kFluid) { p_k0-=1.f; }
        if(t_k1 == kAir) { p+=1.f; } else if(t_k1 == kSolid || t_k1 == kWall) { p-=1.f; } else if(t_k1 >= kFluid) { p_k1-=1.f; }

        float* vals = A.valuesOnRow( n );
        size_t* inds = A.indicesOnRow( n );

        vals[0] = p;
        vals[1] = p_i0;
        vals[2] = p_i1;
        vals[3] = p_j0;
        vals[4] = p_j1;
        vals[5] = p_k0;
        vals[6] = p_k1;

        inds[0] = t_cc-kFluid;
        inds[1] = t_i0 >= kFluid ? t_i0-kFluid : NULL_MAX;
        inds[2] = t_i1 >= kFluid ? t_i1-kFluid : NULL_MAX;
        inds[3] = t_j0 >= kFluid ? t_j0-kFluid : NULL_MAX;
        inds[4] = t_j1 >= kFluid ? t_j1-kFluid : NULL_MAX;
        inds[5] = t_k0 >= kFluid ? t_k0-kFluid : NULL_MAX;
        inds[6] = t_k1 >= kFluid ? t_k1-kFluid : NULL_MAX;

        px[n] = pressField(i,j,k);
        pb[n] = d;
    };

    LaunchCudaDevice( kernel, 0, count );
    SyncCuda();

    // compute proconditioner _M
    //if( usePreconditioner ) BLAS::buildPreconditioner( _A, _M );
}

int
FluidProjection::solve( Vec3fSparseField& veloField, FloatSparseField& pressField, const IdxSparseField& typField )
{
    /* Solve linear-system */
    int iternum(0);
    
    if( usePreconditioner )
    {
        iternum = LinearSystemSolver::pcg( _A, _M, _x, _b, maxIteration );
    }
    else
    {
        iternum = LinearSystemSolver::cg( _A, _x, _b, maxIteration );
    }

    const IndexArray& fluidCellBuffer = _fluidCellBuffer;

    float* pPress = pressField.pointer();
    const float* xArr = _x.pointer();
    const float* bArr = _b.pointer();

    auto kernel_press = [=] BORA_DEVICE ( const size_t n )
    {
        size_t idx = fluidCellBuffer[n];

        size_t t = typField[idx] - kFluid;

        pPress[idx] = xArr[t];
    };

    LaunchCudaDevice( kernel_press, 0, fluidCellBuffer.size() );
    SyncCuda();

    /* Gradient pressure field */
    size_t EI = typField.nx()-1;
    size_t EJ = typField.ny()-1;
    size_t EK = typField.nz()-1;

    auto kernel_neig_press = [=] BORA_DEVICE ( const size_t n )
    {
        if( typField[n] == kAir )
        {
            size_t i,j,k;
            pressField.findIndices( n, i,j,k );

            float press(0.f);
            int count(0);

            if( i > 0  && typField(i-1,j,k) >= kFluid ){ press += pressField(i-1,j,k); count++; }
            if( i < EI && typField(i+1,j,k) >= kFluid ){ press += pressField(i+1,j,k); count++; }
            if( j > 0  && typField(i,j-1,k) >= kFluid ){ press += pressField(i,j-1,k); count++; }
            if( j < EJ && typField(i,j+1,k) >= kFluid ){ press += pressField(i,j+1,k); count++; }
            if( k > 0  && typField(i,j,k-1) >= kFluid ){ press += pressField(i,j,k-1); count++; }
            if( k < EK && typField(i,j,k+1) >= kFluid ){ press += pressField(i,j,k+1); count++; }

            if( count > 0 ) pPress[n] = press/(float)count;
        }
    };

    LaunchCudaDevice( kernel_neig_press, 0, typField.size() );
    SyncCuda();

    Vec3f* pv = veloField.pointer();

    const float wallLv = wallLevel;

    auto kernel = [=] BORA_DEVICE ( const size_t t )
    {
        size_t n = fluidCellBuffer[t];

        size_t i,j,k;
        pressField.findIndices( n, i,j,k );

        size_t wall = (float)j+0.5f <= wallLv+1.f ? (size_t)kSolid : (size_t)kAir;

        const float p = pressField[n];
        
        const size_t t_i0 = i ==  0 ? wall : typField( i-1,j,k,false );
        const size_t t_i1 = i == EI ? wall : typField( i+1,j,k,false );
        const size_t t_j0 = j ==  0 ? wall : typField( i,j-1,k,false );
        const size_t t_j1 = j == EJ ? wall : typField( i,j+1,k,false );
        const size_t t_k0 = k ==  0 ? wall : typField( i,j,k-1,false );
        const size_t t_k1 = k == EK ? wall : typField( i,j,k+1,false );

        float i0_j_k = i ==  0 ? 2.f*p-pressField( i+1,j,k ) : pressField( i-1,j,k );
        float i1_j_k = i == EI ? 2.f*p-pressField( i-1,j,k ) : pressField( i+1,j,k );
        float i_j0_k = j ==  0 ? 2.f*p-pressField( i,j+1,k ) : pressField( i,j-1,k );
        float i_j1_k = j == EJ ? 2.f*p-pressField( i,j-1,k ) : pressField( i,j+1,k );
        float i_j_k0 = k ==  0 ? 2.f*p-pressField( i,j,k+1 ) : pressField( i,j,k-1 );
        float i_j_k1 = k == EK ? 2.f*p-pressField( i,j,k-1 ) : pressField( i,j,k+1 );

        float p_i0 = t_i0 == kSolid ? 2.f*p-i1_j_k : t_i0 == kAir ? -i0_j_k : i0_j_k;
        float p_i1 = t_i1 == kSolid ? 2.f*p-i0_j_k : t_i1 == kAir ? -i1_j_k : i1_j_k;
        float p_j0 = t_j0 == kSolid ? 2.f*p-i_j1_k : t_j0 == kAir ? -i_j0_k : i_j0_k;
        float p_j1 = t_j1 == kSolid ? 2.f*p-i_j0_k : t_j1 == kAir ? -i_j1_k : i_j1_k;
        float p_k0 = t_k0 == kSolid ? 2.f*p-i_j_k1 : t_k0 == kAir ? -i_j_k0 : i_j_k0;
        float p_k1 = t_k1 == kSolid ? 2.f*p-i_j_k0 : t_k1 == kAir ? -i_j_k1 : i_j_k1;

        Vec3f grad( (p_i1-p_i0)*0.5f, (p_j1-p_j0)*0.5f, (p_k1-p_k0)*0.5f );
        pv[n] -= grad;
    };

    LaunchCudaDevice( kernel, 0, fluidCellBuffer.size() );
    SyncCuda();

    return iternum;
}

BORA_NAMESPACE_END

