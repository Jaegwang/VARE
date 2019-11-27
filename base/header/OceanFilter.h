//---------------//
// OceanFilter.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.02.26                               //
//-------------------------------------------------------//

#ifndef _BoraOceanFilter_h_
#define _BoraOceanFilter_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

class OceanFilter
{
    private:

        OceanFilterType type;

        float x0;
        float x1;
        float x2;
        float x3;

    public:

        OceanFilter( const OceanParams& params = OceanParams() )
        {
            OceanFilter::set( params );
        }

        void set( const OceanParams& params )
        {
            type = params.filterType;

            x0  = params.filterSmallWavelength - params.filterSoftWidth;
            x1  = params.filterSmallWavelength;
            x2  = params.filterBigWavelength;
            x3  = params.filterBigWavelength + params.filterSoftWidth;

            switch( type )
            {
                default:
                case kNoFilter:
                {
                    // nothing to do
                }
                break;

                case kSmoothBandPass:
                {
                }
                break;
            }
        }

        float operator()( const float& kMag ) const
        {
            switch( type )
            {
                default:
                case kNoFilter:
                {
                    return 1.f;
                }
                break;

                case kSmoothBandPass:
                {
                    const float x = PI2 / kMag; // wave length
                    return ( SmoothStep(x,x0,x1) - SmoothStep(x,x2,x3) );
                }
                break;
            }
        }
};

BORA_NAMESPACE_END

#endif

