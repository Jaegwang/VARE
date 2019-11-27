//-------------------//
// DenseFieldUtils.h //
//-------------------------------------------------------//
// author: Julie Jang @ Dexter Studios                   //
// last update: 2018.04.12                               //
//-------------------------------------------------------//

#ifndef _BoraDenseFieldUtils_h_
#define _BoraDenseFieldUtils_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

void SetSphere( const Vec3f& center, const float radius, ScalarDenseField& lvs );

bool Gradient( const ScalarDenseField& lvs, VectorDenseField& nrm );
//bool Divergence ( const VectorDenseField& vel, ScalarDenseField& div );
//bool Curl       ( const VectorDenseField& vel, VectorDenseField& crl );

BORA_NAMESPACE_END

#endif

