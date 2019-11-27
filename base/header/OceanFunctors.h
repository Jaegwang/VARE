//-----------------//
// OceanFunctors.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.04.24                               //
//-------------------------------------------------------//

#ifndef _BoraOceanFunctors_h_
#define _BoraOceanFunctors_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

struct OceanFunctor_InitialState
{
    OceanFilter                 filter;
    OceanSpectrum               spectrum;
    OceanRandomDistribution     random;
    OceanDirectionalSpreading   spreading;
    OceanDispersionRelationship dispersion;

    float windAngle = 0.f;
    float PI2_L = 0.f; // = 2 * pi / L (L: domain size)

    float*    Omega = nullptr; // angular frequency by dispersion relationship
    Complexf* H0Pos = nullptr; // waves to the left  in the frequency domain
    Complexf* H0Neg = nullptr; // waves to the right in the frequency domain

    void operator()( const int& index )
    {
        Omega[index] = 0.f;
        H0Pos[index].zeroize();
        H0Neg[index].zeroize();
    }

    void operator()( const Vec2f& k, const int& index )
    {
        const float kMag = k.length();

        random.seed( k );

        const float thetaPos = ATan2( -k.y,  k.x ) + windAngle;
        const float thetaNeg = ATan2(  k.y, -k.x ) + windAngle;

        const float& w = Omega[index] = dispersion.w( kMag );
        const float dwdk = dispersion.dwdk( kMag, w );

        const float S = spectrum( w );

        const float d = ( PI2_L * PI2_L ) * dwdk / kMag;
        const float DeltaSPos = S * spreading( w, thetaPos ) * d;
        const float DeltaSNeg = S * spreading( w, thetaNeg ) * d;

        const float f = filter( kMag );

        const float ampPos = random.nextAmp() * Sqrt( Abs( 2.f * DeltaSPos ) ) * f;
        const float ampNeg = random.nextAmp() * Sqrt( Abs( 2.f * DeltaSNeg ) ) * f;

        const float phasePos = random.nextPhase();
        const float phaseNeg = random.nextPhase();

        H0Pos[index] = ampPos * Complexf( Cos(phasePos), -Sin(phasePos) );
        H0Neg[index] = ampNeg * Complexf( Cos(phaseNeg), -Sin(phaseNeg) );
    }
};

struct OceanFunctor_Dh
{
    // input
    float time;
    float timeScale;
    float loopingDuration; // unit: sec.
    const float*    Omega = nullptr;
    const Complexf* H0Pos = nullptr;
    const Complexf* H0Neg = nullptr;

    // output
    Complexf* HSpec = nullptr;

    void operator()( const int& index )
    {
        HSpec[index].zeroize();
    }

    void operator()( const Vec2f& k, const int& index )
    {
        float w = Omega[index];
        const float wp = PI2 / ( loopingDuration * timeScale );
        w = (int)(w/wp) * wp;

        const float t = time * timeScale;

        const float cos = Cos( w * t );
        const float sin = Sin( w * t );

        const Complexf fwd( cos, -sin ); // forward
        const Complexf bwd( cos,  sin ); // backward

        HSpec[index] = ( H0Pos[index] * fwd ) + ( H0Neg[index] * bwd );
   }
};

struct OceanFunctor_Dxx
{
    // input
    const Complexf* HSpec = nullptr;

    // output
    Complexf* DxxSpec = nullptr;

    void operator()( const int& index )
    {
        DxxSpec[index].zeroize();
    }

    void operator()( const Vec2f& k, const int& index )
    {
        const float kMag = k.length();
        DxxSpec[index] = Complexf( Pow2(k.x)/kMag, 0.f ) * HSpec[index];
    }
};

struct OceanFunctor_Dyy
{
    // input
    const Complexf* HSpec = nullptr;

    // output
    Complexf* DyySpec = nullptr;

    void operator()( const int& index )
    {
        DyySpec[index].zeroize();
    }

    void operator()( const Vec2f &k, const int& index )
    {
        const float kMag = k.length();
        DyySpec[index] = Complexf( Pow2(k.y)/kMag, 0.f ) * HSpec[index];
    }
};

struct OceanFunctor_Dxy
{
    // input
    const Complexf* HSpec = nullptr;

    // output
    Complexf* DxySpec = nullptr;

    void operator()( const int& index )
    {
        DxySpec[index].zeroize();
    }

    void operator()( const Vec2f& k, const int& index )
    {
        const float kMag = k.length();
        DxySpec[index] = Complexf( (k.x*k.y) / kMag, 0.f ) * HSpec[index];
    }
};

struct OceanFunctor_J
{
    // input
    float pinch = 0.f;
    const float* Dxx = nullptr;
    const float* Dyy = nullptr;
    const float* Dxy = nullptr;

    // output
    float* J = nullptr;

    void operator()( const tbb::blocked_range<int> &r ) const
    {
        for( int i=r.begin(); i!=r.end(); ++i )
        {
            const float Jxx = 1.f - pinch * Dxx[i];
            const float Jyy = 1.f - pinch * Dyy[i];
            const float Jxy = -pinch * Dxy[i]; // Jxy = Jyx

            const float A = 0.5f * ( Jxx + Jyy );
            const float B = 0.5f * Sqrt( Pow2( Jxx - Jyy ) + 4.f * Pow2( Jxy ) );

            // J+(=B+A) creates whitecaps at trough.
            // J-(=B-A) creates whitecaps at crest.
            J[i] = B - A;
        }
    }
};

// Note that this is not the derivative.
struct OceanFunctor_Dx
{
    // input
    const Complexf* HSpec = nullptr;

    // output
    Complexf* DxSpec = nullptr;

    void operator()( const int& index )
    {
        DxSpec[index].zeroize();
    }

    void operator()( const Vec2f& k, const int& index )
    {
        const float kMag = k.length();
        DxSpec[index] = Complexf( 0.f, -k.x/kMag ) * HSpec[index];
    }
};

// Note that this is not the derivative.
struct OceanFunctor_Dy
{
    // input
    const Complexf* HSpec = nullptr;

    // output
    Complexf* DySpec = nullptr;

    void operator()( const int& index )
    {
        DySpec[index].zeroize();
    }

    void operator()( const Vec2f& k, const int& index )
    {
        const float kMag = k.length();
        DySpec[index] = Complexf( 0.f, -k.y/kMag ) * HSpec[index];
    }
};

template <typename OceanFunctor>
class ComputeSpectralField
{
    private:

        const OceanFunctor& m_func;

        int    m_resolution;
        int    m_stride;
        float  m_physicalLength;
        float  m_PI2_L;

    public:

        ComputeSpectralField( const OceanFunctor& i_func, int i_resolution, float i_physicalLength, size_t i_grainSize )
        : m_func(i_func)
        {
            const int& N = i_resolution;

            const int Nx = N/2 + 1;
            const int Ny = N;

            m_resolution     = N;
            m_stride         = Nx;
            m_physicalLength = i_physicalLength;
            m_PI2_L          = PI2 / m_physicalLength;

            tbb::parallel_for( tbb::blocked_range2d<int>( 0,Ny,1, 0,Nx,i_grainSize ), *this );
        }

        void operator()( const tbb::blocked_range2d<int>& i_range ) const
        {
            const int& N = m_resolution;

            OceanFunctor func( m_func );

            for( int r=i_range.rows().begin(); r!=i_range.rows().end(); ++r )
            {
                const int j = ( r <= N/2 ) ? r : ( r - N );

                // Add a very small random perturbation for enhancing numerical stability when using looping duration.
                const float kj = ( j * m_PI2_L ) * Rand(j,0.999f,1.001f);

                int index = (r*m_stride) + i_range.cols().begin();

                for( int i=i_range.cols().begin(); i!=i_range.cols().end(); ++i, ++index )
                {
                    if( i==0 && j==0 )
                    {
                        func( index );
                    }
                    else
                    {
                        // Add a very small random perturbation for enhancing numerical stability when using looping duration.
                        const float ki = ( i * m_PI2_L ) * Rand(i+j,0.999f,1.001f);

                        func( Vec2f(ki,kj), index );
                    }
                }
            }
        }
};

struct OceanFunctor_Vertex
{
    // input
    int   N;
    float h;

    float amplitudeGain;
    float pinch;

    float crestGain;
    float crestBias;
    int   crestAccumulation;
    float crestDecay;

    const float* Dh = nullptr;
    const float* Dx = nullptr;
    const float* Dy = nullptr;
    const float* J  = nullptr;

    // output
    Vec3f* GRD = nullptr;
    Vec3f* POS = nullptr;
    Vec3f* NRM = nullptr;
    float* WAV = nullptr;
    float* CRS = nullptr;

    int index( const int& i, const int& j ) const
    {
        return ( i + j*(N+1) );
    }

    Vec3f pointAtIndex( const int& i, const int& j, const int& index ) const
    {
        Vec3f p;

        p.x = ( i * h ) - ( pinch * Dx[index] );
        p.y = amplitudeGain * Dh[index];
        p.z = ( j * h ) - ( pinch * Dy[index] );

        return p;
    }

    void operator()( const tbb::blocked_range2d<int>& range ) const
    {
        for( int j=range.rows().begin(); j!=range.rows().end(); ++j )
        {
            const int j_down   = Wrap( j-1, N );
            const int j_center = Wrap( j,   N );
            const int j_up     = Wrap( j+1, N );

            for( int i=range.cols().begin(); i!=range.cols().end(); ++i )
            {
                const int i_left   = Wrap( i-1, N );
                const int i_center = Wrap( i,   N );
                const int i_right  = Wrap( i+1, N );

                const int idx = index(i,j);

                const int idx_left  = index( i_left,   j_center );
                const int idx_right = index( i_right,  j_center );
                const int idx_down  = index( i_center, j_down   );
                const int idx_up    = index( i_center, j_up     );

                const Vec3f p_left  = pointAtIndex( -1,  0, idx_left  );
                const Vec3f p_right = pointAtIndex( +1,  0, idx_right );
                const Vec3f p_down  = pointAtIndex(  0, -1, idx_down  );
                const Vec3f p_up    = pointAtIndex(  0, +1, idx_up    );

                const Vec3f dpdx = p_right - p_left;
                const Vec3f dpdz = p_up    - p_down;

                Vec3f& p = POS[idx];
                p.x = GRD[idx].x - ( pinch * Dx[idx] );
                p.y = amplitudeGain * Dh[idx];
                p.z = GRD[idx].z - ( pinch * Dy[idx] );

                NRM[idx] = ( dpdz ^ dpdx ).normalize();

                WAV[idx] = amplitudeGain * Dh[idx];

                float& c = CRS[idx];

                if( crestAccumulation )
                {
                    c += Max( ( J[idx] * crestGain ) + crestBias, 0.f );
                    c = Clamp( c, 0.f, 1.f );
                    c *= 1.f - crestDecay;
                }
                else
                {
                    c = ( J[idx] * crestGain ) + crestBias;
                    c = Clamp( c, 0.f, 1.f );
                }
            }
        }
    }
};

struct OceanFunctor_Interpolation
{
    // input
    int   N;
    float L;
    float h;

    float eps;
    float time;
    float flowSpeed;
    Vec3f wind;

    Vec3f* tmpPOS = nullptr;
    Vec3f* tmpNRM = nullptr;
    float* tmpWAV = nullptr;
    float* tmpCRS = nullptr;

    // output
    Vec3f* POS = nullptr;
    Vec3f* NRM = nullptr;
    float* WAV = nullptr;
    float* CRS = nullptr;

    int index( const int& i, const int& j ) const
    {
        return ( i + j*(N+1) );
    }

    void operator()( const tbb::blocked_range2d<int>& range ) const
    {
        for( int j=range.rows().begin(); j!=range.rows().end(); ++j )
        {
            for( int i=range.cols().begin(); i!=range.cols().end(); ++i )
            {
                const int idx = index(i,j);

                const float wx = ( i * h ) - ( time * flowSpeed * wind.x );
                const float wz = ( j * h ) - ( time * flowSpeed * wind.z );

                const float x = Clamp( Wrap( wx, L ), 0.f, L-eps );
                const float z = Clamp( Wrap( wz, L ), 0.f, L-eps );

                const int I = (int)( x / h );
                const int J = (int)( z / h );

                Vec3f origin;
                origin.x = Floor( wx / L ) * L;
                origin.z = Floor( wz / L ) * L;

                const float a = x - (I*h);
                const float b = h - a;
                const float c = z - (J*h);
                const float d = h - c;

                const float hh = h*h;

                const float w0 = ( b * d ) / hh;
                const float w1 = ( a * d ) / hh;
                const float w2 = ( b * c ) / hh;
                const float w3 = ( a * c ) / hh;

                const int v0 = index( I  , J   );
                const int v1 = index( I+1, J   );
                const int v2 = index( I  , J+1 );
                const int v3 = index( I+1, J+1 );

                POS[idx] = w0*tmpPOS[v0] + w1*tmpPOS[v1] + w2*tmpPOS[v2] + w3*tmpPOS[v3] + origin;
                NRM[idx] = w0*tmpNRM[v0] + w1*tmpNRM[v1] + w2*tmpNRM[v2] + w3*tmpNRM[v3];
                WAV[idx] = w0*tmpWAV[v0] + w1*tmpWAV[v1] + w2*tmpWAV[v2] + w3*tmpWAV[v3];
                CRS[idx] = w0*tmpCRS[v0] + w1*tmpCRS[v1] + w2*tmpCRS[v2] + w3*tmpCRS[v3];

                POS[idx].x += ( time * flowSpeed * wind.x );
                POS[idx].z += ( time * flowSpeed * wind.z );
            }
        }
    }
};

BORA_NAMESPACE_END

#endif

