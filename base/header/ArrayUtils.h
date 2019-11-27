//--------------//
// ArrayUtils.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.05.10                               //
//-------------------------------------------------------//

#ifndef _BoraArrayUtils_h_
#define _BoraArrayUtils_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

AABB2f BoundingBox( const Vec2fArray& points );
AABB3f BoundingBox( const Vec3fArray& points );

Vec3f MinMagVector( const Vec3fArray& vectors );
Vec3f MaxMagVector( const Vec3fArray& vectors );

void ScatterOnSphere
(
    // input
    const size_t count,
    const float  radius,
    const Vec3f& center,
    const bool   asAppending,

    // output
    Vec3fArray&  points
);

BORA_NAMESPACE_END

#endif

