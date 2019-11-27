//-------------------//
// DenseFieldUtils.h //
//-------------------------------------------------------//
// author: Julie Jang @ Dexter Studios                   //
// last update: 2018.05.15                               //
//-------------------------------------------------------//

#ifndef _BoraDenseFieldUtils2D_h_
#define _BoraDenseFieldUtils2D_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

void SetCircle( const Vec2f& center/*worldspace*/, const float radius/*worldspace*/, ScalarDenseField2D& lvs/*voxelSpace*/ );

//void SetDensitySource( const Vec2f& center/*worldspace*/, const float radius/*worldspace*/, ScalarDenseField2D& dst/*voxelSpace*/, const float density, const Vec3f buoyancy );

bool Gradient( VectorDenseField2D& v, const ScalarDenseField2D& s );

bool Divergence( ScalarDenseField2D& s, const VectorDenseField2D& v );

bool Curl( ScalarDenseField2D& s, const VectorDenseField2D& v );

BORA_NAMESPACE_END

#endif

