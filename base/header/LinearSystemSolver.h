//----------------------//
// LinearSystemSolver.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.07.03                               //
//-------------------------------------------------------//

#pragma once
#include <Bora.h>

BORA_NAMESPACE_BEGIN

class LinearSystemSolver
{
    public:

        static int cg ( const FloatSparseMatrix& A, FloatArray& x, FloatArray& b, const int max=200 );
        static int pcg( const FloatSparseMatrix& A, FloatArray& M, FloatArray& x, FloatArray& b, const int max=200, const bool pre=true );

};

BORA_NAMESPACE_END

