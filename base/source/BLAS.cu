//----------//
// BLAS.cpp //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.07.03                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN namespace BLAS { // Basic Linear Algebra Subprogram

void mul( const FloatSparseMatrix& a, const FloatArray& b, FloatArray& c )
{
    const size_t rows = a.rows();
    const size_t cols = a.columns();

    if( rows != b.size() ) return;
    c.initialize( rows, b.memorySpace() );

    float* cp = c.pointer();

    auto kernel = [=] BORA_DEVICE ( const size_t r )
    {
        float* vals = a.valuesOnRow( r );
        size_t* inds = a.indicesOnRow( r );

        float acc = 0.f;

        for( size_t c=0; c<cols; ++ c )
        {
            size_t n = inds[c];
            if( n != NULL_MAX ) acc += vals[c] * b[n];
        }

        cp[r] = acc;
    };

    LaunchCudaDevice( kernel, 0, rows );
    SyncCuda();
}

void add( const FloatArray& a, const FloatArray& b, FloatArray& c )
{
    c.initialize( a.size(), a.memorySpace() );
    float* cp = c.pointer();

    auto kernel = [=] BORA_DEVICE ( const size_t n )
    {
        cp[n] = a[n] + b[n];
    };

    LaunchCudaDevice( kernel, 0, a.size() );
    SyncCuda();
}

void sub( const FloatArray& a, const FloatArray& b, FloatArray& c )
{
    c.initialize( a.size(), a.memorySpace() );
    float* cp = c.pointer();    
    
    auto kernel = [=] BORA_DEVICE ( const size_t n )
    {
        cp[n] = a[n] - b[n];
    };
    
    LaunchCudaDevice( kernel, 0, a.size() );
    SyncCuda();
}

void mul( const FloatArray& a, const FloatArray& b, FloatArray& c )
{
    c.initialize( a.size(), a.memorySpace() );
    float* cp = c.pointer();

    auto kernel = [=] BORA_DEVICE ( const size_t n )
    {   
        cp[n] = a[n] * b[n];
    };

    LaunchCudaDevice( kernel, 0, a.size() );
    SyncCuda();
}

void mul( const FloatArray& a, const float b, FloatArray& c )
{
    c.initialize( a.size(), a.memorySpace() );
    float* cp = c.pointer();
    
    auto kernel = [=] BORA_DEVICE ( const size_t n )
    {
        cp[n] = a[n] * b;
    };
    
    LaunchCudaDevice( kernel, 0, a.size() );
    SyncCuda();
}

float dot( const FloatArray& a, const FloatArray& b )
{
    thrust::device_ptr<float> d_a( a.pointer() );
    thrust::device_ptr<float> d_b( b.pointer() );

    return thrust::inner_product( d_a, d_a+a.size(), d_b, 0.f );
}

void equ( const FloatArray& a, FloatArray& c )
{
    c.initialize( a.size(), a.memorySpace() );
    for( size_t n=0; n<a.size(); ++n )
    {
        c[n] = a[n];
    }
}

float len( const FloatArray& a )
{
    float l(0.f);
    for( size_t n=0; n<a.size(); ++n )
    {
        l += a[n] * a[n];
    }
    return sqrt( l );
}

void addmul( const FloatArray& a, const float& alpha, const FloatArray& p, FloatArray& x )
{
    float* px = x.pointer();

    auto kernel = [=] BORA_DEVICE ( const size_t n )
    {
        px[n] = a[n] + (alpha * p[n]);
    };

    LaunchCudaDevice( kernel, 0, a.size() );
    SyncCuda();
}

void submul( const FloatArray& a, const float& alpha, const FloatArray& p, FloatArray& x )
{
    float* px = x.pointer();
    
    auto kernel = [=] BORA_DEVICE ( const size_t n )
    {
        px[n] = a[n] - (alpha * p[n]);
    };

    LaunchCudaDevice( kernel, 0, a.size() );
    SyncCuda();
}

void addmul( const FloatArray& a, const FloatSparseMatrix& A, const FloatArray& p, FloatArray& x )
{
    const size_t rows = A.rows();
    const size_t cols = A.columns();

    if( rows != p.size() ) return;
    x.initialize( rows, p.memorySpace() );

    float* px = x.pointer();

    auto kernel = [=] BORA_DEVICE ( const size_t r )
    {
        float* vals = A.valuesOnRow( r );
        size_t* inds = A.indicesOnRow( r );

        float acc = 0.f;

        for( size_t c=0; c<cols; ++ c )
        {
            size_t n = inds[c];
            if( n != NULL_MAX ) acc += vals[c] * p[n];
        }

        px[r] = a[r] + acc;
    };

    LaunchCudaDevice( kernel, 0, rows );
    SyncCuda();
}

void submul( const FloatArray& a, const FloatSparseMatrix& A, const FloatArray& p, FloatArray& x )
{
    const size_t rows = A.rows();
    const size_t cols = A.columns();

    if( rows != p.size() ) return;
    x.initialize( rows, p.memorySpace() );

    float* px = x.pointer();

    auto kernel = [=] BORA_DEVICE ( const size_t r )
    {
        float* vals = A.valuesOnRow( r );
        size_t* inds = A.indicesOnRow( r );

        float acc = 0.f;

        for( size_t c=0; c<cols; ++ c )
        {
            size_t n = inds[c];
            if( n != NULL_MAX ) acc += vals[c] * p[n];
        }

        px[r] = a[r] - acc;
    };

    LaunchCudaDevice( kernel, 0, rows );
    SyncCuda();
}

void buildPreconditioner( const SparseMatrix<float>& A, FloatArray& M )
{
    M.setValueAll( 0.f );
    float* pm = M.pointer();

    const float _micParam = 0.97f;    

    for( size_t n=0; n<A.rows(); ++n )    
    {
        const float* a_values = A.valuesOnRow( n );
        const size_t* a_indices = A.indicesOnRow( n );

        const size_t& a_i0jk = a_indices[1];
        const size_t& a_ij0k = a_indices[3];
        const size_t& a_ijk0 = a_indices[5];

        float e = a_values[0];
        float e2( 0.f );

        float alpha = e;

        if( a_i0jk != NULL_MAX )
        {
            const float* V = A.valuesOnRow(a_i0jk);
            
            e -= Pow2( V[2] * M[a_i0jk] );
            e2 += V[2] * ( V[4]+V[6] ) * Pow2( M[a_i0jk] );
        }        

        if( a_ij0k != NULL_MAX )
        {
            const float* V = A.valuesOnRow(a_ij0k);

            e -= Pow2( V[4] * M[a_ij0k] );
            e2 += V[4] * ( V[2]+V[6] ) * Pow2( M[a_ij0k] );
        }

        if( a_ijk0 != NULL_MAX )
        {
            const float* V = A.valuesOnRow(a_ijk0);

            e -= Pow2( V[6] * M[a_ijk0] );
            e2 += V[6] * ( V[2]+V[4] ) * Pow2( M[a_ijk0] );
        }

        e -= _micParam * e2;

        if( e < alpha * 0.25f ) e = alpha;

        pm[n] = 1.f / sqrt( e + EPSILON );
    };    
}

void applyPreconditioner( const SparseMatrix<float>& A, const FloatArray& M, const FloatArray& r, FloatArray& q, FloatArray& z )
{
    z.initialize( r.size(), r.memorySpace() );
    q.initialize( r.size(), r.memorySpace() );

    for( size_t n=0; n<r.size(); ++n )
    {        
        const float* a_values = A.valuesOnRow( n );
        const size_t* a_indices = A.indicesOnRow( n );

        size_t a_i0jk = a_indices[1];
        size_t a_ij0k = a_indices[3];
        size_t a_ijk0 = a_indices[5];

        float t = r[n];

        if( a_i0jk != NULL_MAX )
        {
            const float* V = A.valuesOnRow(a_i0jk);
            t -= V[2] * M[ a_i0jk ] * q[ a_i0jk ];
        }        

        if( a_ij0k != NULL_MAX )
        {
            const float* V = A.valuesOnRow(a_ij0k);
            t -= V[4] * M[ a_ij0k ] * q[ a_ij0k ];
        }

        if( a_ijk0 != NULL_MAX )
        {
            const float* V = A.valuesOnRow(a_ijk0);
            t -= V[6] * M[ a_ijk0 ] * q[ a_ijk0 ];
        }

        q[n] = t * M[n];
    }

    for( size_t l=0; l<r.size(); ++l )
    {
        size_t n = r.size()-1 - l;

        const float* a_values = A.valuesOnRow( n );
        const size_t* a_indices = A.indicesOnRow( n );

        size_t a_i1jk = a_indices[2];
        size_t a_ij1k = a_indices[4];
        size_t a_ijk1 = a_indices[6];

        float t = q[n];

        if( a_i1jk != NULL_MAX )
        { 
            t -= a_values[2] * M[n] * z[a_i1jk];
        }

        if( a_ij1k != NULL_MAX )
        {
            t -= a_values[4] * M[n] * z[a_ij1k];
        }

        if( a_ijk1 != NULL_MAX )
        {
            t -= a_values[6] * M[n] * z[a_ijk1];
        }
        
        z[n] = t * M[n];
    }    
}

} BORA_NAMESPACE_END

