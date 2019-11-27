//------------//
// TBBUtils.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.01.09                               //
//-------------------------------------------------------//

#ifndef _BoraTBBUtils_h_
#define _BoraTBBUtils_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

//////////////////////////////////////////////////////////////////////////////////
// print

static tbb::mutex bora_tbb_printMutex;

#define MUTEX_PRINT(x)                                 \
do                                                     \
{                                                      \
    tbb::mutex::scoped_lock lock(bora_tbb_printMutex); \
    std::cout << x << std::endl;                       \
} while (0)

//////////////////////////////////////////////////////////////////////////////////
// min & max

template <typename TT>
struct TBBMinMaxFunctor
{
    TT min = (TT)( +INFINITE );
    TT max = (TT)( -INFINITE );

    TBBMinMaxFunctor()
    {
        // nothing to do
    }

    TBBMinMaxFunctor( const TBBMinMaxFunctor&, tbb::split )
    {
        // nothing to do
    }

    void operator()( const tbb::blocked_range<const TT*>& r )
    {
        TT tmpMin = min;
        TT tmpMax = max;

        for( const TT* itr=r.begin(); itr!=r.end(); ++itr )
        {
            const TT& v = *itr;

            tmpMin = Min( tmpMin, v );
            tmpMax = Max( tmpMax, v );
        }

        min = tmpMin;
        max = tmpMax;
    }

    void join( const TBBMinMaxFunctor& rhs )
    {
        min = Min( min, rhs.min );
        max = Max( max, rhs.max );
    }
};

template <typename TT>
struct TBBMinMaxCalculator
{
    TT min = (TT)0;
    TT max = (TT)0;

    TBBMinMaxCalculator( const TT* v, const size_t n )
    {
        TBBMinMaxFunctor<TT> F;
        tbb::parallel_reduce( tbb::blocked_range<const TT*>(v,v+n), F );

        min = F.min;
        max = F.max;
    }
};

//////////////////////////////////////////////////////////////////////////////////
// sum

template <typename TT>
struct TBBSumFunctor
{
    TT sum = (TT)0;

    TBBSumFunctor()
    {
        // nothing to do
    }

    TBBSumFunctor( const TBBSumFunctor& i_s, tbb::split )
    {
        // nothing to do
    }

    void operator()( const tbb::blocked_range<const TT*>& r )
    {
        TT tmp = sum;

        for( const TT* itr=r.begin(); itr!=r.end(); ++itr )
        {
            const TT& v = *itr;

            tmp += v;
        }

        sum = tmp;
    }

    void join( const TBBSumFunctor& rhs )
    {
        sum += rhs.sum;
    }
};

template <typename TT>
struct TBBSumCalculator
{
    TT sum = (TT)0;

    TBBSumCalculator( const TT* v, const size_t n )
    {
        TBBSumFunctor<TT> F;
        tbb::parallel_reduce( tbb::blocked_range<const TT*>(v,v+n), F );

        sum = F.sum;
    }
};

BORA_NAMESPACE_END

#endif

