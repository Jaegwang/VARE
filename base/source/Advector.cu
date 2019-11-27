//--------------//
// Advector.cpp //
//-------------------------------------------------------//
// author: Julie Jang @ Dexter Studios                   //
// last update: 2018.04.16                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

Advector::Advector()
{
    // nothing to do
}

void
Advector::set( float cfl_number, int minSubSteps, int maxSubSteps, float Dt, AdvectionScheme scheme )
{
	_cfl		 = cfl_number;
	_minSubSteps = minSubSteps;
	_maxSubSteps = maxSubSteps;
	_Dt			 = Dt;
	_advScheme	 = scheme;	
}

int
Advector::advect( ScalarDenseField& s, const VectorDenseField& v )
{
	int maxMagIdx;
	float maxVelMagnitude = v.maxMagnitude( maxMagIdx );

	float dt = _cfl/maxVelMagnitude;
	int substeps = Abs(_Dt/dt);
	substeps = Clamp( substeps, _minSubSteps, _maxSubSteps );
	dt = _Dt/(float)substeps;

    Nx = s.nx();
    Ny = s.ny();
    Nz = s.nz();

    if( (Grid)_tmpScalarField != (Grid)s )
    {
        _tmpScalarField.initialize( s.getGrid(), kUnified ); // memory allocation
    }
    _tmpScalarField.swap( s );
    switch( _advScheme )
    {
        default:
        case AdvectionScheme::kLinear:     { byLinear     ( s, v, substeps, dt ); break; }
        case AdvectionScheme::kRK2:        { byRK2        ( s, v, substeps, dt ); break; }
        case AdvectionScheme::kRK3:        { byRK3        ( s, v, substeps, dt ); break; }
        case AdvectionScheme::kRK4:        { byRK4        ( s, v, substeps, dt ); break; }
        case AdvectionScheme::kMacCormack: { byMacCormack ( s, v, substeps, dt ); break; }
        case AdvectionScheme::kBFECC:      { byBFECC      ( s, v, substeps, dt ); break; }
    }
	return substeps;
}

int
Advector::advect( VectorDenseField& s, const VectorDenseField& v )
{
	int maxMagIdx;
	float maxVelMagnitude = v.maxMagnitude( maxMagIdx );

	float dt = _cfl/maxVelMagnitude;
	int substeps = Abs(_Dt/dt);
	substeps = Clamp( substeps, _minSubSteps, _maxSubSteps );
	dt = _Dt/(float)substeps;

    Nx = s.nx();
    Ny = s.ny();
    Nz = s.nz();

    if( (Grid)_tmpVectorField != (Grid)s )
    {
        _tmpVectorField.initialize( s.getGrid() ); // memory allocation
    }
	_tmpVectorField.swap( s );
    switch( _advScheme )
    {
        default:
        case AdvectionScheme::kLinear:     { byLinear     ( s, v, substeps, dt ); break; }
        case AdvectionScheme::kRK2:        { byRK2        ( s, v, substeps, dt ); break; }
        case AdvectionScheme::kRK3:        { byRK3        ( s, v, substeps, dt ); break; }
        case AdvectionScheme::kRK4:        { byRK4        ( s, v, substeps, dt ); break; }
        case AdvectionScheme::kMacCormack: { byMacCormack ( s, v, substeps, dt ); break; }
        case AdvectionScheme::kBFECC:      { byBFECC      ( s, v, substeps, dt ); break; }
    }
	return substeps;
}

void 
Advector::byLinear( ScalarDenseField& s, const VectorDenseField& v, int substeps, float dt )
{
	float* s_ptr = s.pointer();
	const ScalarDenseField& tmpScalarField = _tmpScalarField;

	auto kernel = [=] BORA_FUNC_QUAL ( size_t n )
	{
		size_t i,j,k;
        s.cellIndices( n, i, j, k );

        Vec3f vel;
        Vec3f q = s.cellCenter( i, j, k );
		for( int x=0; x<substeps; ++x )
		{
			vel = v.lerp( q );
        	q   = q - ( dt * vel );
		}
        s_ptr[n] = tmpScalarField.lerp( q );
	};

	LaunchCudaDevice( kernel, 0, s.numCells() );
	SyncCuda();
}

void 
Advector::byLinear( VectorDenseField& s, const VectorDenseField& v, int substeps, float dt )
{
	Vec3f* s_ptr = s.pointer();
	const VectorDenseField& tmpVectorField = _tmpVectorField;

	auto kernel = [=] BORA_FUNC_QUAL ( size_t n )
	{
		size_t i,j,k;
        s.cellIndices( n, i, j, k );

        Vec3f vel;
        Vec3f q = s.cellCenter( i, j, k );
		for( int x=0; x<substeps; ++x )
		{
			vel = v.lerp( q );
        	q   = q - ( dt * vel );
		}
        s_ptr[n] = tmpVectorField.lerp( q );
	};

	LaunchCudaDevice( kernel, 0, s.numCells() );
	SyncCuda();

}

void
Advector::byRK2( ScalarDenseField& s, const VectorDenseField& v, int substeps, float dt )
{
	float* s_ptr = s.pointer();
	const ScalarDenseField& tmpScalarField = _tmpScalarField;

	auto kernel = [=] BORA_FUNC_QUAL ( size_t n )
	{
		size_t i,j,k;
        s.cellIndices( n, i, j, k );

        Vec3f v1, v2, q1, q2;
        q2 = s.cellCenter( i, j, k );
		for( int x=0; x<substeps; ++x )
		{
			v1 = v.lerp( q2 );

        	q1 = q2 - ( dt * 0.5f * v1 );
        	v2 = v.lerp( q1 );

        	q2 = q2 - ( dt * v2 );
		}
        s_ptr[n] = tmpScalarField.lerp( q2 );

	};

	LaunchCudaDevice( kernel, 0, s.numCells() );
	SyncCuda();

}

void
Advector::byRK2( VectorDenseField& s, const VectorDenseField& v, int substeps, float dt )
{
	Vec3f* s_ptr = s.pointer();
	const VectorDenseField& tmpVectorField = _tmpVectorField;

	auto kernel = [=] BORA_FUNC_QUAL ( size_t n )
	{
		size_t i,j,k;
        s.cellIndices( n, i, j, k );

        Vec3f v1, v2, q1, q2;
        q2 = s.cellCenter( i, j, k );
		for( int x=0; x<substeps; ++x )
		{
			v1 = v.lerp( q2 );

        	q1 = q2 - ( dt * 0.5f * v1 );
        	v2 = v.lerp( q1 );

        	q2 = q2 - ( dt * v2 );
		}
        s_ptr[n] = tmpVectorField.lerp( q2 );

	};

	LaunchCudaDevice( kernel, 0, s.numCells() );
	SyncCuda();
}

void
Advector::byRK3( ScalarDenseField& s, const VectorDenseField& v, int substeps, float dt )
{
	float* s_ptr = s.pointer();
	const ScalarDenseField& tmpScalarField = _tmpScalarField;

	auto kernel = [=] BORA_FUNC_QUAL ( size_t n )
	{
		size_t i,j,k;
        s.cellIndices( n, i, j, k );

        Vec3f v1, v2, v3, m, q1, q2, q3;
        q3 = s.cellCenter( i, j, k );
		for( int x=0; x<substeps; ++x )
		{
			v1 = v.lerp( q3 );

        	q1 = q3 - ( dt * 0.5f * v1 );
        	v2 = v.lerp( q1 );

        	q2 = q3 + ( dt * v1 ) - ( dt * 2.f * v2 );
        	v3 = v.lerp( q2 );

        	m = ( v1 + ( 4.f * v2 ) + v3 ) / 6.f;

        	q3 = q3 - ( dt * m );
		}
        s_ptr[n] = tmpScalarField.lerp( q3 );

	};

	LaunchCudaDevice( kernel, 0, s.numCells() );
	SyncCuda();

}

void
Advector::byRK3( VectorDenseField& s, const VectorDenseField& v, int substeps, float dt )
{
	Vec3f* s_ptr = s.pointer();
	const VectorDenseField& tmpVectorField = _tmpVectorField;

	auto kernel = [=] BORA_FUNC_QUAL ( size_t n )
	{
		size_t i,j,k;
        s.cellIndices( n, i, j, k );

        Vec3f v1, v2, v3, m, q1, q2, q3;
        q3 = s.cellCenter( i, j, k );
		for( int x=0; x<substeps; ++x )
		{
			v1 = v.lerp( q3 );

        	q1 = q3 - ( dt * 0.5f * v1 );
        	v2 = v.lerp( q1 );

        	q2 = q3 + ( dt * v1 ) - ( dt * 2.f * v2 );
        	v3 = v.lerp( q2 );

        	m = ( v1 + ( 4.f * v2 ) + v3 ) / 6.f;

        	q3 = q3 - ( dt * m );
		}
        s_ptr[n] = tmpVectorField.lerp( q3 );

	};

	LaunchCudaDevice( kernel, 0, s.numCells() );
	SyncCuda();
}

void
Advector::byRK4( ScalarDenseField& s, const VectorDenseField& v, int substeps, float dt )
{
	float* s_ptr = s.pointer();
	const ScalarDenseField& tmpScalarField = _tmpScalarField;

	auto kernel = [=] BORA_FUNC_QUAL ( size_t n )
	{
		size_t i,j,k;
        s.cellIndices( n, i, j, k );

        Vec3f v1, v2, v3, v4, m, q1, q2, q3, q4;
        q4 = s.cellCenter( i, j, k );
		for( int x=0; x<substeps; ++x )
		{
			v1 = v.lerp( q4 );

	        q1 = q4 - ( dt * 0.5f * v1 );
	        v2 = v.lerp(q1);
	
	        q2 = q4 - ( dt * 0.5f * v2 );
	        v3 = v.lerp( q2 );
	
	        q3 = q4 - ( dt * v3 );
	        v4 = v.lerp( q3 );
	
	        m = (v1/6.f) + (v2/3.f) + (v3/3.f) + (v4/6.f);
	
	        q4 = q4 - ( dt * m );
		}
        s_ptr[n] = tmpScalarField.lerp( q4 );

	};

	LaunchCudaDevice( kernel, 0, s.numCells() );
	SyncCuda();
}

void
Advector::byRK4( VectorDenseField& s, const VectorDenseField& v, int substeps, float dt )
{
	Vec3f* s_ptr = s.pointer();
	const VectorDenseField& tmpVectorField = _tmpVectorField;

	auto kernel = [=] BORA_FUNC_QUAL ( size_t n )
	{
		size_t i,j,k;
        s.cellIndices( n, i, j, k );

        Vec3f v1, v2, v3, v4, m, q1, q2, q3, q4;
        q4 = s.cellCenter( i, j, k );
		for( int x=0; x<substeps; ++x )
		{
			v1 = v.lerp( q4 );

	        q1 = q4 - ( dt * 0.5f * v1 );
	        v2 = v.lerp(q1);
	
	        q2 = q4 - ( dt * 0.5f * v2 );
	        v3 = v.lerp( q2 );
	
	        q3 = q4 - ( dt * v3 );
	        v4 = v.lerp( q3 );
	
	        m = (v1/6.f) + (v2/3.f) + (v3/3.f) + (v4/6.f);
	
	        q4 = q4 - ( dt * m );
		}
        s_ptr[n] = tmpVectorField.lerp( q4 );

	};

	LaunchCudaDevice( kernel, 0, s.numCells() );
	SyncCuda();


}

void
Advector::byMacCormack( ScalarDenseField& s, const VectorDenseField& v, int substeps, float dt )
{
	float* s_ptr = s.pointer();
	const ScalarDenseField& tmpScalarField = _tmpScalarField;

	auto kernel = [=] BORA_FUNC_QUAL ( size_t n )
	{
		size_t i,j,k;
        s.cellIndices( n, i, j, k );

        Vec3f v1, v2, q1, q2, q3;
        q3 = s.cellCenter( i, j, k );
		for( int x=0; x<substeps; ++x )
		{
			v1 = v.lerp( q3 );

	        q1 = q3 - ( dt * v1 );
	        v2 = v.lerp( q1 );
	
	        q2 = q1 + ( dt * v2 );
	
	        q3 = q1 + ( 0.5f * ( q3 - q2 ) );
		}
	    s_ptr[n] = tmpScalarField.lerp( q3 );

	};

	LaunchCudaDevice( kernel, 0, s.numCells() );
	SyncCuda();


}

void
Advector::byMacCormack( VectorDenseField& s, const VectorDenseField& v, int substeps, float dt )
{
	Vec3f* s_ptr = s.pointer();
	const VectorDenseField& tmpVectorField = _tmpVectorField;

	auto kernel = [=] BORA_FUNC_QUAL ( size_t n )
	{
		size_t i,j,k;
        s.cellIndices( n, i, j, k );

        Vec3f v1, v2, q1, q2, q3;
        q3 = s.cellCenter( i, j, k );
		for( int x=0; x<substeps; ++x )
		{
			v1 = v.lerp( q3 );

	        q1 = q3 - ( dt * v1 );
	        v2 = v.lerp( q1 );
	
	        q2 = q1 + ( dt * v2 );
	
	        q3 = q1 + ( 0.5f * ( q3 - q2 ) );
		}
        s_ptr[n] = tmpVectorField.lerp( q3 );

	};

	LaunchCudaDevice( kernel, 0, s.numCells() );
	SyncCuda();

}

void
Advector::byBFECC( ScalarDenseField& s, const VectorDenseField& v, int substeps, float dt )
{
	float* s_ptr = s.pointer();
	const ScalarDenseField& tmpScalarField = _tmpScalarField;
	size_t& nx = Nx; size_t& ny = Ny; size_t& nz = Nz;

	auto kernel = [=] BORA_FUNC_QUAL ( size_t n )
	{
		size_t i,j,k;
        s.cellIndices( n, i, j, k );

        Vec3f v1, v2, v3, q1, q2, q3, q4;
        q4 = s.cellCenter( i, j, k );
		for( int x=0; x<substeps; ++x )
		{
			v1 = v.lerp( q4 );

	        q1 = q4 - ( dt * v1 );
	
	        q1.x = Clamp( q1.x, 0.f, nx-EPSILON );
	        q1.y = Clamp( q1.y, 0.f, ny-EPSILON );
	        q1.z = Clamp( q1.z, 0.f, nz-EPSILON );
	
	        v2 = v.lerp( q1 );
	
	        q2 = q1 + ( dt * v2 );
	
	        q2.x = Clamp( q2.x, 0.f, nx-EPSILON );
	        q2.y = Clamp( q2.y, 0.f, ny-EPSILON );
	        q2.z = Clamp( q2.z, 0.f, nz-EPSILON );
	
	        q3 = q4 + ( 0.5f * ( q4 - q2 ) );
	        v3 = v.lerp( q3 );
	
	        q4 = q3 - ( dt * v3 );
		}
        s_ptr[n] = tmpScalarField.lerp( q4 );

	};

	LaunchCudaDevice( kernel, 0, s.numCells() );
	SyncCuda();

}

void
Advector::byBFECC( VectorDenseField& s, const VectorDenseField& v, int substeps, float dt )
{

	Vec3f* s_ptr = s.pointer();
	const VectorDenseField& tmpVectorField = _tmpVectorField;
	size_t& nx = Nx; size_t& ny = Ny; size_t& nz = Nz;

	auto kernel = [=] BORA_FUNC_QUAL ( size_t n )
	{
		size_t i,j,k;
        s.cellIndices( n, i, j, k );

        Vec3f v1, v2, v3, q1, q2, q3, q4;
        q4 = s.cellCenter( i, j, k );
		for( int x=0; x<substeps; ++x )
		{
			v1 = v.lerp( q4 );

	        q1 = q4 - ( dt * v1 );
	
	        q1.x = Clamp( q1.x, 0.f, nx-EPSILON );
	        q1.y = Clamp( q1.y, 0.f, ny-EPSILON );
	        q1.z = Clamp( q1.z, 0.f, nz-EPSILON );
	
	        v2 = v.lerp( q1 );
	
	        q2 = q1 + ( dt * v2 );
	
	        q2.x = Clamp( q2.x, 0.f, nx-EPSILON );
	        q2.y = Clamp( q2.y, 0.f, ny-EPSILON );
	        q2.z = Clamp( q2.z, 0.f, nz-EPSILON );
	
	        q3 = q4 + ( 0.5f * ( q4 - q2 ) );
	        v3 = v.lerp( q3 );
	
	        q4 = q3 - ( dt * v3 );
		}

        s_ptr[n] = tmpVectorField.lerp( q4 );

	};

	LaunchCudaDevice( kernel, 0, s.numCells() );
	SyncCuda();

}

BORA_NAMESPACE_END

