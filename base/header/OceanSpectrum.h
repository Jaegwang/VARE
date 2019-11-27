//-----------------//
// OceanSpectrum.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.02.26                               //
//-------------------------------------------------------//

#ifndef _BoraOceanSpectrum_h_
#define _BoraOceanSpectrum_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

class OceanSpectrum
{
    private:

        OceanSpectrumType type;

        float g;           // gravity
        float U;           // wind speed
        float wp;          // the peak omega
        float sqrt_h_g;    // sqrt(h/g)
        float alpha, beta; // coefficients for Alpha-Beta Spectrum
        float gamma = 3.3f;

    public:

        OceanSpectrum( const OceanParams& params = OceanParams() )
        {
            OceanSpectrum::set( params );
        }

        void set( const OceanParams& params )
        {
            type = params.spectrumType;

            g  = params.gravity;
            U  = params.windSpeed;

            const float F = params.fetch * 1000.f; // fetch (1000: km -> m)

            switch( type )
            {
                case kPhillips:
                {
                    wp = 0.855f * g / U; // (eqn. 19)

                    alpha = 1e-4f * g*g;
                    beta = Pow4( g / U );
                }
                break;

                case kPiersonMoskowitz:
                {
                    wp = 0.855f * g / U; // (eqn. 19)

                    alpha = 8.1e-3f;
                    beta  = 0.74f;
                }
                break;

                case kJONSWAP:
                {
                    wp = 22.f * Pow( (g*g)/(U*F), 1.f/3.f ); // https://www.wikiwaves.org/Ocean-WaveSpectra

                    alpha = 0.076f * Pow( (U*U)/(F*g), 0.22f );
                    beta  = 1.25f;
                }
                break;

                default:
                case kTMA:
                {
                    wp = 22.f * Pow( (g*g)/(U*F), 1.f/3.f ); // https://www.wikiwaves.org/Ocean-WaveSpectra

                    alpha = 0.076f * Pow( (U*U)/(F*g), 0.22f );
                    beta  = 1.25f;

                    const float h = params.depth;
                    const float g = params.gravity;

                    sqrt_h_g = Sqrt( h / g );
                }
                break;
            };
        }

        float operator()( const float& w ) const
        {
            switch( type )
            {
                case kPhillips:
                {
                    return AlphaBetaSpectrum( alpha, beta, g, w, wp );
                }
                break;

                case kPiersonMoskowitz:
                {
                    return AlphaBetaSpectrum( alpha, beta, g, w, wp );
                }
                break;

                case kJONSWAP:
                {
                    const float sigma = ( w <= wp ) ? 0.07f : 0.09f;
                    const float b = Exp( -0.5f * Pow2( (w-wp) / (sigma*wp) ) );

                    // extra peak enhancement factor
                    const float rb = Pow( gamma, b );

                    return ( rb * AlphaBetaSpectrum( alpha, beta, g, w, wp ) );
                }
                break;

                default:
                case kTMA:
                {
                    const float sigma = ( w <= wp ) ? 0.07f : 0.09f;
                    const float b = Exp( -0.5f * Pow2( (w-wp) / (sigma*wp) ) );

                    // extra peak enhancement factor
                    const float rb = Pow( gamma, b );

                    const float wh = w * sqrt_h_g;

                    // Kitaigorodskii Depth Attenuation Function
                    const float PHI = SmoothStep( wh, 0.f, 2.f );

                    return ( rb * AlphaBetaSpectrum( alpha, beta, g, w, wp ) * PHI );
                }
                break;
            }
        }

    private:

        // The generalized A,B Ppectrum (eqn. 20)
        //
        // g : gravitational acceleration
        // w : angular frequency (= omega)
        // wp: peak (=modal) angular frequency (in the paper, omega_p = omega_m = omega_0)
        //
        float AlphaBetaSpectrum( const float& alpha, const float& beta, const float& g, const float& w, const float& wp ) const
        {
            const float A = alpha * (g*g) / (w*w*w*w*w);
            const float B = beta * (wp*wp*wp*wp);

            return ( A * Exp( -B / (w*w*w*w) ) );
        }
};

BORA_NAMESPACE_END

#endif

