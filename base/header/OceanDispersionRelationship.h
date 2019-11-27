//-------------------------------//
// OceanDispersionRelationship.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.02.26                               //
//-------------------------------------------------------//

#ifndef _BoraOceanDispersionRelationship_h_
#define _BoraOceanDispersionRelationship_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

class OceanDispersionRelationship
{
    private:

        OceanDispersionRelationshipType type;

        float g;     // gravitational acceleration
        float h;     // water depth
        float sigma; // surface tension coefficient
        float rho;   // water density

    public:

        OceanDispersionRelationship( const OceanParams& params = OceanParams() )
        {
            OceanDispersionRelationship::set( params );
        }

        void set( const OceanParams& params )
        {
            type = params.dispersionRelationshipType;

            switch( type )
            {
                default:
                case kDeepWater:
                {
                    g = params.gravity;
                }
                break;

                case kFiniteDepthWater:
                {
                    g = params.gravity;
                    h = params.depth;
                }
                break;

                case kVerySmallDepthWater:
                {
                    g = params.gravity;
                    h = params.depth;

                    sigma = params.surfaceTension;
                    rho   = params.density;
                }
                break;
            }
        }

        // k (=kMag) -> w
        float w( const float& k ) const
        {
            switch( type )
            {
                default:
                case kDeepWater:
                {
                    return Sqrt( Abs( g*k ) );
                }
                break;

                case kFiniteDepthWater:
                {
                    return Sqrt( Abs( g*k * Tanh(k*h) ) );
                }
                break;

                case kVerySmallDepthWater:
                {
                    return Sqrt( Abs( ( ( g*k ) + ( sigma/rho * Pow3(k) ) ) * Tanh(h*k) ) );
                }
                break;
            }
        }

        // k (=kMag) & w -> dw/dk
        float dwdk( const float& k, const float& w ) const
        {
            switch( type )
            {
                default:
                case kDeepWater:
                {
                    return ( g / ( 2.f * w ) );
                }
                break;

                case kFiniteDepthWater:
                {
                    const float kh = h * k;
                    return ( ( g * ( Tanh(kh) + kh / Pow2(Cosh(kh)) ) ) / ( 2.f * w ) );
                }
                break;

                case kVerySmallDepthWater:
                {
                    const float kh    = h * k;
                    const float k2s   = Pow2(k) * sigma/rho;
                    const float gpk2s = g + k2s;
                    const float numer = ( ( gpk2s + k2s + k2s ) * Tanh(kh) ) + (kh * gpk2s / Pow2( Cosh(kh) ) );
                    return ( Abs(numer) / ( 2.f * w ) );
                }
                break;
            }
        }
};

BORA_NAMESPACE_END

#endif

