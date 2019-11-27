//--------//
// BLAS.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.07.03                               //
//-------------------------------------------------------//

#pragma once
#include <Bora.h>

BORA_NAMESPACE_BEGIN

namespace BLAS { // Basic Linear Algebra Subprogram

// add sub mul div...
void  mul( const FloatSparseMatrix& a, const FloatArray& b, FloatArray& c );
void  add( const FloatArray& a, const FloatArray& b, FloatArray& c );
void  sub( const FloatArray& a, const FloatArray& b, FloatArray& c );
void  mul( const FloatArray& a, const FloatArray& b, FloatArray& c );
void  mul( const FloatArray& a, const float b, FloatArray& c );
float dot( const FloatArray& a, const FloatArray& b );
void  equ( const FloatArray& a, FloatArray& c );
float len( const FloatArray& a );

void addmul( const FloatArray& a, const float& alpha, const FloatArray& p, FloatArray& x );
void submul( const FloatArray& a, const float& alpha, const FloatArray& p, FloatArray& x );
void addmul( const FloatArray& a, const FloatSparseMatrix& A, const FloatArray& p, FloatArray& x );
void submul( const FloatArray& a, const FloatSparseMatrix& A, const FloatArray& p, FloatArray& x );

void buildPreconditioner( const SparseMatrix<float>& A, FloatArray& M );
void applyPreconditioner( const SparseMatrix<float>& A, const FloatArray& M, const FloatArray& r, FloatArray& q, FloatArray& z );

}

BORA_NAMESPACE_END

