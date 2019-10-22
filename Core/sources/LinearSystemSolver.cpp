
#include <VARE.h>

VARE_NAMESPACE_BEGIN

int LinearSystemSolver::cg( const FloatSparseMatrix& A, FloatArray& x, FloatArray& b, const int max )
{
    FloatArray r, p, Ap;

    BLAS::submul( b, A, x, r ); // r = b - A * x;

    BLAS::equ( r, p ); // p = r;

    float rsold = BLAS::dot( r, r ); // r' * r;

    if( sqrt( rsold ) < 1e-30f ) return 0;

    int iter = b.size();
    int count(0);
    for( int i=0; i<iter+5; ++i, ++count )
    {
        BLAS::mul( A, p, Ap );

        float alpha = rsold / BLAS::dot( p, Ap );

        BLAS::addmul( x, alpha, p, x );
        
        BLAS::submul( r, alpha, Ap, r );

        float rsnew = BLAS::dot( r, r );

        if( sqrt( rsnew ) < 1e-3f ) break;

        BLAS::addmul( r, rsnew/rsold, p, p );

        rsold = rsnew;
        if( i > iter ) std::cout<<"CG-Error"<<std::endl;

        if( count > max ) break;
    }

    return count;
}

int LinearSystemSolver::pcg( const FloatSparseMatrix& A, FloatArray& M, FloatArray& x, FloatArray& b, const int max, const bool pre )
{
    FloatArray r, q, t, d, s;

    BLAS::mul( A, x, t );
    BLAS::sub( b, t, r ); // r = b - A * x;

    if( pre ) BLAS::applyPreconditioner( A, M, r, t, d );
    else BLAS::equ( r, d );

    float alpha_new = BLAS::dot( r,d );
    size_t i(0);

    const size_t maxIter = Min( b.size(), (size_t)max );

    while( alpha_new > 1e-3f && i < maxIter ) 
    {
        BLAS::mul( A, d, q ); // q = Ad

        float alpha = alpha_new / BLAS::dot( d, q );

        BLAS::mul( d, alpha, t );
        BLAS::add( x, t, x );

        if( i % 50 == 0 )
        {
            BLAS::mul( A, x, t );
            BLAS::sub( b, t, r );
        }
        else
        {
            BLAS::mul( q, alpha, t );
            BLAS::sub( r, t, r );
        }

        if( pre ) BLAS::applyPreconditioner( A, M, r, t, s );
        else BLAS::equ( r, s );

        float alpha_old = alpha_new;

        alpha_new = BLAS::dot( r, s );

        float beta = alpha_new / alpha_old;


        BLAS::mul( d, beta, t );
        BLAS::add( s, t, d ); // d = s + beta * d

        i++;
    }

    return i; 
}

VARE_NAMESPACE_END
