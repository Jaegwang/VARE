//--------------//
// Advector2D.cpp //
//-------------------------------------------------------//
// author: Julie Jang @ Dexter Studios                   //
// last update: 2018.04.25                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

Advector2D::Advector2D()
{
    // nothing to do
}

void
Advector2D::set( float cfl_number, int minSubSteps, int maxSubSteps, float Dt, AdvectionScheme scheme )
{
	_cfl		 = cfl_number;
	_minSubSteps = minSubSteps;
	_maxSubSteps = maxSubSteps;
	_Dt			 = Dt;
	_advScheme	 = scheme;

}

int
Advector2D::advect( ScalarDenseField2D& s, const VectorDenseField2D& v )
{
	int maxMagIdx;
	float maxVelMagnitude = v.maxMagnitude( maxMagIdx );

	float dt = _cfl/maxVelMagnitude;
	int substeps = Abs(_Dt/dt);
	substeps = Clamp( substeps, _minSubSteps, _maxSubSteps );
	dt = _Dt/(float)substeps;

    Nx = s.nx();
    Ny = s.ny();

    if( (Grid2D)_tmpScalarField != (Grid2D)s )
    {
        _tmpScalarField.initialize( s.getGrid2D(), kUnified ); // memory allocation
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
Advector2D::advect( VectorDenseField2D& s, const VectorDenseField2D& v )
{
	int maxMagIdx;
	float maxVelMagnitude = v.maxMagnitude( maxMagIdx );

	float dt = _cfl/maxVelMagnitude;
	int substeps = Abs(_Dt/dt);
	substeps = Clamp( substeps, _minSubSteps, _maxSubSteps );
	dt = _Dt/(float)substeps;

    Nx = s.nx();
    Ny = s.ny();

    if( (Grid2D)_tmpVectorField != (Grid2D)s )
    {
        _tmpVectorField.initialize( s.getGrid2D() ); // memory allocation
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
Advector2D::byLinear( ScalarDenseField2D& s, const VectorDenseField2D& v, int substeps, float dt )
{
	float* s_ptr = s.pointer();
	const ScalarDenseField2D& tmpScalarField = _tmpScalarField;

	auto kernel = [=] BORA_FUNC_QUAL ( size_t n )
	{
		size_t i,j;
        s.cellIndices( n, i, j );

        Vec2f vel;
        Vec2f q = s.cellCenter( i, j );
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
Advector2D::byLinear( VectorDenseField2D& s, const VectorDenseField2D& v, int substeps, float dt )
{
	Vec2f* s_ptr = s.pointer();
	const VectorDenseField2D& tmpVectorField = _tmpVectorField;

	auto kernel = [=] BORA_FUNC_QUAL ( size_t n )
	{
		size_t i,j;
        s.cellIndices( n, i, j );

        Vec2f vel;
        Vec2f q = s.cellCenter( i, j );
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
Advector2D::byRK2( ScalarDenseField2D& s, const VectorDenseField2D& v, int substeps, float dt )
{
	float* s_ptr = s.pointer();
	const ScalarDenseField2D& tmpScalarField = _tmpScalarField;

	auto kernel = [=] BORA_FUNC_QUAL ( size_t n )
	{
		size_t i,j;
        s.cellIndices( n, i, j );

        Vec2f v1, v2, q1, q2;
        q2 = s.cellCenter( i, j );
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
Advector2D::byRK2( VectorDenseField2D& s, const VectorDenseField2D& v, int substeps, float dt )
{
	Vec2f* s_ptr = s.pointer();
	const VectorDenseField2D& tmpVectorField = _tmpVectorField;

	auto kernel = [=] BORA_FUNC_QUAL ( size_t n )
	{
		size_t i,j;
        s.cellIndices( n, i, j );

        Vec2f v1, v2, q1, q2;
        q2 = s.cellCenter( i, j );
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
Advector2D::byRK3( ScalarDenseField2D& s, const VectorDenseField2D& v, int substeps, float dt )
{
	float* s_ptr = s.pointer();
	const ScalarDenseField2D& tmpScalarField = _tmpScalarField;

	auto kernel = [=] BORA_FUNC_QUAL ( size_t n )
	{
		size_t i,j;
        s.cellIndices( n, i, j );

        Vec2f v1, v2, v3, m, q1, q2, q3;
        q3 = s.cellCenter( i, j );
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
Advector2D::byRK3( VectorDenseField2D& s, const VectorDenseField2D& v, int substeps, float dt )
{
	Vec2f* s_ptr = s.pointer();
	const VectorDenseField2D& tmpVectorField = _tmpVectorField;

	auto kernel = [=] BORA_FUNC_QUAL ( size_t n )
	{
		size_t i,j;
        s.cellIndices( n, i, j );

        Vec2f v1, v2, v3, m, q1, q2, q3;
        q3 = s.cellCenter( i, j );
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
Advector2D::byRK4( ScalarDenseField2D& s, const VectorDenseField2D& v, int substeps, float dt )
{
	float* s_ptr = s.pointer();
	const ScalarDenseField2D& tmpScalarField = _tmpScalarField;

	auto kernel = [=] BORA_FUNC_QUAL ( size_t n )
	{
		size_t i,j;
        s.cellIndices( n, i, j );

        Vec2f v1, v2, v3, v4, m, q1, q2, q3, q4;
        q4 = s.cellCenter( i, j );
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
Advector2D::byRK4( VectorDenseField2D& s, const VectorDenseField2D& v, int substeps, float dt )
{
	Vec2f* s_ptr = s.pointer();
	const VectorDenseField2D& tmpVectorField = _tmpVectorField;

	auto kernel = [=] BORA_FUNC_QUAL ( size_t n )
	{
		size_t i,j;
        s.cellIndices( n, i, j );

        Vec2f v1, v2, v3, v4, m, q1, q2, q3, q4;
        q4 = s.cellCenter( i, j );
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
Advector2D::byMacCormack( ScalarDenseField2D& s, const VectorDenseField2D& v, int substeps, float dt )
{
	float* s_ptr = s.pointer();
	const ScalarDenseField2D& tmpScalarField = _tmpScalarField;

	auto kernel = [=] BORA_FUNC_QUAL ( size_t n )
	{
		size_t i,j;
        s.cellIndices( n, i, j );

        Vec2f v1, v2, q1, q2, q3;
        q3 = s.cellCenter( i, j );
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
Advector2D::byMacCormack( VectorDenseField2D& s, const VectorDenseField2D& v, int substeps, float dt )
{
	Vec2f* s_ptr = s.pointer();
	const VectorDenseField2D& tmpVectorField = _tmpVectorField;

	auto kernel = [=] BORA_FUNC_QUAL ( size_t n )
	{
		size_t i,j;
        s.cellIndices( n, i, j );

        Vec2f v1, v2, q1, q2, q3;
        q3 = s.cellCenter( i, j );
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
Advector2D::byBFECC( ScalarDenseField2D& s, const VectorDenseField2D& v, int substeps, float dt )
{
	float* s_ptr = s.pointer();
	const ScalarDenseField2D& tmpScalarField = _tmpScalarField;
	size_t& nx = Nx; size_t& ny = Ny; 

	auto kernel = [=] BORA_FUNC_QUAL ( size_t n )
	{
		size_t i,j;
        s.cellIndices( n, i, j );

        Vec2f v1, v2, v3, q1, q2, q3, q4;
        q4 = s.cellCenter( i, j );
		for( int x=0; x<substeps; ++x )
		{
			v1 = v.lerp( q4 );

	        q1 = q4 - ( dt * v1 );
	
	        q1.x = Clamp( q1.x, 0.f, nx-EPSILON );
	        q1.y = Clamp( q1.y, 0.f, ny-EPSILON );
	
	        v2 = v.lerp( q1 );
	
	        q2 = q1 + ( dt * v2 );
	
	        q2.x = Clamp( q2.x, 0.f, nx-EPSILON );
	        q2.y = Clamp( q2.y, 0.f, ny-EPSILON );
	
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
Advector2D::byBFECC( VectorDenseField2D& s, const VectorDenseField2D& v, int substeps, float dt )
{

	Vec2f* s_ptr = s.pointer();
	const VectorDenseField2D& tmpVectorField = _tmpVectorField;
	size_t& nx = Nx; size_t& ny = Ny; 

	auto kernel = [=] BORA_FUNC_QUAL ( size_t n )
	{
		size_t i,j;
        s.cellIndices( n, i, j );

        Vec2f v1, v2, v3, q1, q2, q3, q4;
        q4 = s.cellCenter( i, j );
		for( int x=0; x<substeps; ++x )
		{
			v1 = v.lerp( q4 );

	        q1 = q4 - ( dt * v1 );
	
	        q1.x = Clamp( q1.x, 0.f, nx-EPSILON );
	        q1.y = Clamp( q1.y, 0.f, ny-EPSILON );
	
	        v2 = v.lerp( q1 );
	
	        q2 = q1 + ( dt * v2 );
	
	        q2.x = Clamp( q2.x, 0.f, nx-EPSILON );
	        q2.y = Clamp( q2.y, 0.f, ny-EPSILON );
	
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

