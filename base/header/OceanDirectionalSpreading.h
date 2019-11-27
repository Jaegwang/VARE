//-----------------------------//
// OceanDirectionalSpreading.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.02.26                               //
//-------------------------------------------------------//

#ifndef _BoraOceanDirectionalSpreading_h_
#define _BoraOceanDirectionalSpreading_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

class OceanDirectionalSpreading
{
    private:

        OceanDirectionalSpreadingType type;

        float g;     // gravity
        float U;     // wind speed
        float F;     // fetch length
        float swell; // swell coefficient (xi in the paper)
        float wp;    // modal angular frequency
        float sp;    // model shape

    public:

        OceanDirectionalSpreading( const OceanParams& params = OceanParams() )
        {
            OceanDirectionalSpreading::set( params );
        }

        void set( const OceanParams& params )
        {
            type = params.directionalSpreadingType;

            swell = Clamp( params.swell, 0.f, 1.f );

            g = params.gravity;
            U = params.windSpeed;
            F = params.fetch; // Don't multiply 1,000 for here. (ad hoc)

            wp = PI2 * 3.5f * (g/U) * Pow( (g*F)/(U*U), -0.33f ); // JONSWAP formula
            sp = 11.5f * Pow( wp*U/g, -2.5f );
        }

        float operator()( const float& w, const float& theta ) const
        {
            switch( type )
            {
                case kMitsuyasu:
                {
                    float s = 0.f;
                    {
                        if( w <= wp ) { s = sp * Pow( w/wp,  5.0f ); }
                        else          { s = sp * Pow( w/wp, -2.5f ); }
                    }

                    const float xi = ( 16.f * Tanh( wp/w ) * Pow2( swell ) );

                    const float factor_A = Pow( 2.f, 2.f * s - 1.f ) / PI;
                    const float factor_B = Pow2( TGamma( s + 1.f ) ) / TGamma( 2.f * s + 1.f );
                    const float factor_C = Pow( Abs( Cos( 0.5f * theta ) ), 2*(s+xi) );

                    return ( factor_A * factor_B * factor_C );
                }
                break;

                default:
                case kHasselmann:
                {
                    float s = 0.f;
                    {
                        if( w <= wp ) { s = 6.97f * Pow( w/wp, 4.06f ); }
                        else          { s = 9.77f * Pow( w/wp, -2.33f - ( 1.45f * ( g / wp - 1.17f ) ) ); }
                    }

                    const float xi = ( 16.f * Tanh( wp/w ) * Pow2( swell ) );

                    const float factor_A = Pow( 2.f, 2.f * s - 1.f ) / PI;
                    const float factor_B = Pow2( TGamma( s + 1.f ) ) / TGamma( 2.f * s + 1.f );
                    const float factor_C = Pow( Abs( Cos( 0.5f * theta ) ), 2*(s+xi) );

                    return ( factor_A * factor_B * factor_C );
                }
                break;
            }
        }
};

BORA_NAMESPACE_END

#endif

