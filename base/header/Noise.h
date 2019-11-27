//---------//
// Noise.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2018.12.04                               //
//-------------------------------------------------------//

#pragma once

#include <Bora.h>

BORA_NAMESPACE_BEGIN

class Noise
{
    public:

        Vec4f offset=Vec4f(0.f);
        Vec4f frequency=Vec4f(1.f);
        float scale=1.f;
        Vec3f add=Vec3f(0.f);

    public:

        BORA_FUNC_QUAL Noise()
        {
        }

        BORA_FUNC_QUAL
        float perlin( const float x, const float y, const float z, const float w=0.f, const int oct=1 ) const
        {
            float sum(0.f);
            Vec4f f = frequency;
            float s = scale;            

            for( int n=0; n<oct; ++n )
            {
                sum += glm::perlin( glm::vec4( x*f.x+offset.x, y*f.y+offset.y, z*f.z+offset.z, w*f.w+offset.w ) ) * s;

                f *= 2.0f;
                s *= 0.5f;
            }

            return sum;
        }

        BORA_FUNC_QUAL
        float simplex( const float x, const float y, const float z, const float w=0.f, const int oct=1 ) const
        {
            float sum(0.f);
            Vec4f f = frequency;
            float s = scale;

            for( int n=0; n<oct; ++n )
            {
                sum += glm::simplex( glm::vec4( x*f.x+offset.x, y*f.y+offset.y, z*f.z+offset.z, w*f.w+offset.w ) ) * s;

                f *= 2.0f;
                s *= 0.5f;
            }

            return sum;
        }

        BORA_FUNC_QUAL
        float perlin( const Vec3f& v, const int oct=1 ) const
        {
            return perlin( v.x, v.y, v.z, 0.f, oct );
        }

        BORA_FUNC_QUAL
        float perlin( const Vec4f& v, const int oct=1 ) const
        {
            return perlin( v.x, v.y, v.z, v.w, oct );
        }

        BORA_FUNC_QUAL
        Vec3f perlin3f( const Vec3f& v, const int oct=1 ) const
        {
            return Vec3f( perlin( v+Vec3f(59375.f,29548.f,43923.f), oct ),
                          perlin( v+Vec3f(66632.f,37593.f,19385.f), oct ),
                          perlin( v+Vec3f(95734.f,53842.f,92643.f), oct ) ) + add;
        }

        BORA_FUNC_QUAL
        Vec3f perlin3f( const Vec4f& v, const int oct=1 ) const
        {
            return Vec3f( perlin( v+Vec4f(59375.f,54892.f,19334.f,92643.f), oct ),
                          perlin( v+Vec4f(66632.f,74662.f,98822.f,33289.f), oct ),
                          perlin( v+Vec4f(95734.f,10382.f,33391.f,11822.f), oct ) ) + add;
        }

        BORA_FUNC_QUAL
        float simplex( const Vec3f& v, const int oct=1 ) const 
        {
            return simplex( v.x, v.y, v.z, 0.f, oct );
        }

        BORA_FUNC_QUAL
        float simplex( const Vec4f& v, const int oct=1 ) const 
        {
            return simplex( v.x, v.y, v.z, v.w, oct );
        }

        BORA_FUNC_QUAL
        Vec3f simplex3f( const Vec3f& v, const int oct=1 ) const
        {
            return Vec3f( simplex( v+Vec3f(59375.f,29548.f,43923.f), oct ),
                          simplex( v+Vec3f(66632.f,37593.f,19385.f), oct ),
                          simplex( v+Vec3f(95734.f,53842.f,92643.f), oct ) ) + add;
        }

        BORA_FUNC_QUAL
        Vec3f simplex3f( const Vec4f& v, const int oct=1 ) const
        {
            return Vec3f( simplex( v+Vec4f(59375.f,54892.f,19334.f,92643.f), oct ),
                          simplex( v+Vec4f(66632.f,74662.f,98822.f,33289.f), oct ),
                          simplex( v+Vec4f(95734.f,10382.f,33391.f,11822.f), oct ) ) + add;
        }

        BORA_FUNC_QUAL
        Vec3f curl( const float i, const float j, const float k, const float n=0.f, const int oct=1 ) const
        {
            const Vec3f x0 = simplex3f( Vec4f(i-1.f, j, k, n), oct );
            const Vec3f x1 = simplex3f( Vec4f(i+1.f, j, k, n), oct );
            const Vec3f y0 = simplex3f( Vec4f(i, j-1.f, k, n), oct );
            const Vec3f y1 = simplex3f( Vec4f(i, j+1.f, k, n), oct );
            const Vec3f z0 = simplex3f( Vec4f(i, j, k-1.f, n), oct );
            const Vec3f z1 = simplex3f( Vec4f(i, j, k+1.f, n), oct );

            Vec3f c;
            c.x = (y1.z-y0.z)*0.5f - (z1.y-z0.y)*0.5f;
            c.y = (z1.x-z0.x)*0.5f - (x1.z-x0.z)*0.5f;
            c.z = (x1.y-x0.y)*0.5f - (y1.x-y0.x)*0.5f;

            return c + add;
        }

        BORA_FUNC_QUAL
        Vec3f curl( const Vec3f& p, const float n=0.f, const int oct=1 ) const
        {
            return curl( p.x, p.y, p.z, n, oct );
        }
};

BORA_NAMESPACE_END

