//------------//
// OceanFFT.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.02.26                               //
//-------------------------------------------------------//

#ifndef _BoraOceanFFT_h_
#define _BoraOceanFFT_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

struct CopyWrappedBorder
{
    float* Data;
    int N;

    void operator()( const tbb::blocked_range<int>& i ) const
    {
        const int stride = N+1;

        for( int j=i.begin(); j!=i.end(); ++j )
        {
            if( j == N )
            {
                std::copy( Data, Data+stride, Data+(stride*N) );
            }
            else
            {
                const int rowBeginIndex = j*stride;
                Data[rowBeginIndex+N] = Data[rowBeginIndex];
            }
        }
    }
};

class FFT
{
    private:

        int m_resolution;
        fftwf_plan m_fftPlan;

    public:

        FFT()
        {
        }

        FFT( ComplexOceanSpectralField& input, RealOceanSpatialField& output, int numThreads=-1 )
        {
            FFT::initialize( input, output, numThreads );
        }

        void initialize( ComplexOceanSpectralField& input, RealOceanSpatialField& output, int numThreads=-1 )
        {
            const int N = m_resolution = input.ny();

            const int maxNumThreads = (int)std::thread::hardware_concurrency();
            numThreads = (numThreads<=0) ? maxNumThreads : Min( numThreads, maxNumThreads );

            fftwf_init_threads();
            fftwf_plan_with_nthreads( numThreads );

            const int rank = 2;

            fftwf_iodim dims[2];
            {
                dims[0].n  = N;
                dims[0].is = (N/2)+1;
                dims[0].os = N+1;

                dims[1].n  = N;
                dims[1].is = 1;
                dims[1].os = 1;
            }

            const int howMany = 1;

            fftwf_iodim howManyDims[1];
            {
                howManyDims[0].n = 1;
                howManyDims[0].is = ((N/2)+1)*N;
                howManyDims[0].os = (N+1)*(N+1);
            }

            Complexf* in = input.pointer();
            float* out = output.pointer();

            m_fftPlan = fftwf_plan_guru_dft_c2r( rank, dims, howMany, howManyDims, reinterpret_cast<fftwf_complex*>(in), out, FFTW_ESTIMATE | FFTW_DESTROY_INPUT );
        }

        ~FFT()
        {
            if( m_fftPlan )
            {
                fftwf_destroy_plan( m_fftPlan );
                m_fftPlan = nullptr;
            }

            fftwf_cleanup_threads();
            fftwf_cleanup();
        }

        void execute
        (
            ComplexOceanSpectralField& input,
            RealOceanSpatialField&     output
        )
        {
            const int& N = m_resolution;

            Complexf* in = input.pointer();
            float* out = output.pointer();

            fftwf_execute_dft_c2r( m_fftPlan, reinterpret_cast<fftwf_complex*>(in), out );

            // Fill in the repeated border.
            {
                CopyWrappedBorder F;
                F.Data = output.pointer();
                F.N = N;
                tbb::parallel_for( tbb::blocked_range<int>(0,N+1), F );
            }
        }
};

BORA_NAMESPACE_END

#endif

