//----------//
// Kernel.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2017.10.30                               //
//-------------------------------------------------------//

#ifndef _BoraKernel_h_
#define _BoraKernel_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

BORA_FUNC_QUAL
inline float KernelCubicBSpline( float x )
{
    float h = Abs( x );
  
    if( h >= 0.f && h < 1.f )      return h*h*h*0.5f - x*x + 2.f/3.f;
    else if( h >= 1.f && h < 2.f ) return -1.f/6.f*h*h*h + x*x - 2.f*h + 4.f/3.f;
    else                           return 0.f;
}

BORA_FUNC_QUAL
inline float KernelGradCubicBSpline( float x )
{
    float h = Abs( x );

    if( x <= -2.f ) return 0.f;
    if( x <= -1.f ) return -(-0.5f*h*h + 2.f*h - 2.f);
    if( x <=  0.f ) return -(1.5f*h*h - 2.f*h);
    if( x <=  1.f ) return +(1.5f*h*h - 2.f*h);
    if( x <=  2.f ) return +(-0.5f*h*h + 2.f*h - 2.f);
    else            return 0.f;
}

BORA_FUNC_QUAL
inline float KernelQuadBSpline( float x )
{
    if( x < -1.5f )       return 0.f;
    else if( x <= -0.5f ) return 0.5f*x*x + 1.5f*x + 9.f/8.f;
    else if( x <=  0.5f ) return -x*x + 3.f/4.f;
    else if( x <=  1.5f ) return 0.5f*x*x - 1.5f*x + 9.f/8.f;
    else                  return 0.f;
}

BORA_FUNC_QUAL
inline float KernelGradQuadBSpline( float x )
{
    if( x < -1.5f )       return 0.f;
    else if( x <= -0.5f ) return x + 1.5f;
    else if( x <=  0.5f ) return -2.f*x;
    else if( x <=  1.5f ) return x - 1.5f;
    else                  return 0.f;    
}

BORA_FUNC_QUAL
inline void ScatterPointsFromPrimitive( const Vec3f* vertices, const Vec3f* velocities, const int v_count, const float h, Vec3fArray& posArr, Vec3fArray& velArr )
{
    if( v_count == 1 )
    { // point
        posArr.append( vertices[0] );
        velArr.append( velocities[0] );
    }
    else if( v_count == 2 )
    { // line
        int n = vertices[0].distanceTo(vertices[1])/h + 1;

        for( int i=0; i<=n; ++i )
        {
            float w = (float)i/(float)n;
            Vec3f pos = vertices[0]*w + vertices[1]*(1.f-w);
            Vec3f vel = velocities[0]*w + velocities[1]*(1.f-w);

            posArr.append( pos );
            velArr.append( vel );                        
        }
    }
    else if( v_count == 3 )
    { // triangle
        Vec3f center = (vertices[0]+vertices[1]+vertices[2])*0.333f;

        int n0 = vertices[0].distanceTo(center)*2.f/h + 1;
        int n1 = vertices[1].distanceTo(center)*2.f/h + 1;
        int n2 = vertices[2].distanceTo(center)*2.f/h + 1;

        for( int k=0; k<=n2; ++k )
        for( int j=0; j<=n1; ++j )
        for( int i=1; i<=n0; ++i )
        {
            float w0 = (float)i/(float)n0;
            float w1 = (float)j/(float)n1;
            float w2 = (float)k/(float)n2;
            float tot = w0+w1+w2;

            w0 /= tot;
            w1 /= tot;
            w2 /= tot;

            Vec3f pos = vertices[0]*w0 + vertices[1]*w1 + vertices[2]*w2;
            Vec3f vel = velocities[0]*w0 + velocities[1]*w1 + velocities[2]*w2;                    
        
            posArr.append( pos );
            velArr.append( vel );
        }
    }
    else if( v_count == 4 )
    { // quad
        Vec3f center = (vertices[0]+vertices[1]+vertices[2]+vertices[3])*0.25f;

        int n0 = vertices[0].distanceTo(center)*2.f/h + 1;
        int n1 = vertices[1].distanceTo(center)*2.f/h + 1;
        int n2 = vertices[2].distanceTo(center)*2.f/h + 1;
        int n3 = vertices[3].distanceTo(center)*2.f/h + 1;

        for( int l=0; l<=n3; ++l )
        for( int k=0; k<=n2; ++k )
        for( int j=0; j<=n1; ++j )
        for( int i=1; i<=n0; ++i )
        {
            float w0 = (float)i/(float)n0;
            float w1 = (float)j/(float)n1;
            float w2 = (float)k/(float)n2;
            float w3 = (float)l/(float)n3;
            float tot = w0+w1+w2+w3;

            w0 /= tot;
            w1 /= tot;
            w2 /= tot;
            w3 /= tot;

            Vec3f pos = vertices[0]*w0 + vertices[1]*w1 + vertices[2]*w2 + vertices[3]*w3;
            Vec3f vel = velocities[0]*w0 + velocities[1]*w1 + velocities[2]*w2 + velocities[3]*w3;
        
            posArr.append( pos );
            velArr.append( vel );
        }
    }

}

BORA_NAMESPACE_END

#endif

