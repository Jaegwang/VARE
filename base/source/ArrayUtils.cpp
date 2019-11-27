//----------------//
// ArrayUtils.cpp //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.05.10                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

AABB2f BoundingBox( const Vec2fArray& points )
{
    AABB2f aabb;

    const size_t n = points.size();

    for( size_t i=0; i<n; ++i )
    {
        aabb.expand( points[i] );
    }

    return aabb;
}

AABB3f BoundingBox( const Vec3fArray& points )
{
    AABB3f aabb;

    const size_t n = points.size();

    for( size_t i=0; i<n; ++i )
    {
        aabb.expand( points[i] );
    }

    return aabb;
}

Vec3f MinMagVector( const Vec3fArray& vectors )
{
    float ret = FLT_MAX;

    const size_t n = vectors.size();

    float mag = 0.f;

    for( size_t i=0; i<n; ++i )
    {
        mag = vectors[i].squaredLength();

        if( mag < ret ) { ret = mag; }
    }

    return ret;
}

Vec3f MaxMagVector( const Vec3fArray& vectors )
{
    float ret = 0.f;

    const size_t n = vectors.size();

    float mag = 0.f;

    for( size_t i=0; i<n; ++i )
    {
        mag = vectors[i].squaredLength();

        if( mag > ret ) { ret = mag; }
    }

    return ret;
}

void ScatterOnSphere
(
    // input
    const size_t count,
    const float  radius,
    const Vec3f& center,
    const bool   asAppending,

    // output
    Vec3fArray&  points
)
{
    if( count == 0 ) { return; }
	if( asAppending == false ) { points.clear(); }

	const size_t oldCount = points.size();
	points.resize( oldCount + count );
	const size_t newCount = points.size();

	for( size_t i=oldCount; i<newCount; ++i )
    {
        const float u = Rand( i+137 );
        const float v = Rand( i+737 );

        const float theta = PI2 * u;
        const float phi = ACos( 2.f*v - 1.f );

        points[i].set
        (
            center.x + radius * ( sin(phi) * cos(theta) ),
            center.y + radius * ( sin(phi) * sin(theta) ),
            center.z + radius * ( cos(phi)              )
        );
    }
}

BORA_NAMESPACE_END

