//---------------------------//
// OceanRandomDistribution.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.02.26                               //
//-------------------------------------------------------//

#ifndef _BoraOceanRandomDistribution_h_
#define _BoraOceanRandomDistribution_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

class OceanRandomDistribution
{
    private:

        OceanRandomType type;

        static const uint32_t RANDSEED_P1 = 73856093;
        static const uint32_t RANDSEED_P2 = 19349663;
        static const uint32_t RANDSEED_P3 = 83492791;

        std::uniform_real_distribution<float> m_uniform;
        std::normal_distribution<float>       m_gaussian;
        std::lognormal_distribution<float>    m_logNormal;

        uint_fast32_t m_seed;
        std::minstd_rand m_engine;

    public:

        OceanRandomDistribution( const OceanParams& params = OceanParams() )
        : m_uniform   ( 0.f, PI  )
        , m_gaussian  ( 0.f, 1.f )
        , m_logNormal ( 0.f, 1.f )
        {
            OceanRandomDistribution::set( params );
        }

        void set( const OceanParams& params )
        {
            type = params.randomType;

            m_seed = params.randomSeed;
            m_engine = std::minstd_rand( m_seed );

            OceanRandomDistribution::seed( m_seed );
        }

        void seed( const uint_fast32_t i_seed )
        {
            m_engine.seed( i_seed + m_seed );
        }

        void seed( const Vec2f& k )
        {
            m_engine.seed( uint32_t(k[0]*10000*RANDSEED_P1) ^ uint32_t(k[1]*10000*RANDSEED_P2) ^ (m_seed*RANDSEED_P3) );
        }

        float nextPhase()
        {
            return m_uniform( m_engine );
        }

        float nextAmp()
        {
            switch( type )
            {
                default:
                case kGaussian:
                {
                    return m_gaussian( m_engine );
                }
                break;

                case kLogNormal:
                {
                    return m_logNormal( m_engine );
                }
                break;
            }
        }
};

BORA_NAMESPACE_END

#endif

