
#pragma once
#include <VARE.h>

VARE_NAMESPACE_BEGIN

class LinearSystemSolver
{
    public:

        static int cg ( const FloatSparseMatrix& A, FloatArray& x, FloatArray& b, const int max=200 );
        static int pcg( const FloatSparseMatrix& A, FloatArray& M, FloatArray& x, FloatArray& b, const int max=200, const bool pre=true );

};

VARE_NAMESPACE_END

