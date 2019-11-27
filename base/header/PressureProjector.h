//---------------------//
// PressureProjector.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
//         Julie Jang @ Dexter Studios                   //
// last update: 2018.05.15                               //
//-------------------------------------------------------//

#ifndef _BoraPressureProjector_h_
#define _BoraPressureProjector_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

template <typename T>
class PressureProjector
{
    private:

        Array<T> y;

        FivePointsLaplacian A;

        ScalarDenseField2D &prs;
        VectorDenseField2D &vel;
        ScalarDenseField2D &div;

    public:

        size_t maxIterations = 0;
        ProjectionScheme scheme = kJacobi;

    public:

        PressureProjector
        (
            const ScalarDenseField2D& pressure
            const VectorDenseField2D& velocity
            const ScalarDenseField2D& divergence
        )
        : prs(pressure), vel(velocity), div(divergence)
        {
        }

        void compute();

    private:

        void compute()
        {
            Divergence( vel, div );

            A.build( vel );

            switch( scheme )
            {
                default:
                case kJacobi:
                {
                    doJacobiIteration( A, x, b, maxIterations );
                    break;
                }

            }

            reviseVelocity();
        }

        void doJacobiIteration( const FivePointsLaplacian<T>& A, Array<T>& x, const Array<T>& b, const size_t numIterations )
        {
            x.zeroize(); // initial guess

            y.initialize( x.size(), x.memorySpace() );

            const size_t nx = A.nx();
            const size_t ny = A.ny();
            const size_t dimension = A.dimension();

            for( size_t itr=0; itr<numIterations; ++itr )
            {
                for( size_t iRow=0; iRow<dimension; ++iRow )
                {
                    // y <- A, x, b

                    T& yVal = y[iRow];

                    yVal = b[iRow];
                    yVal -= A.south(iRow)  * x[iRow-Nx];
                    yVal -= A.west(iRow)   * x[iRow-1 ];
                    yVal -= A.center(iRow) * x[iRow   ];
                    yVal -= A.east(iRow)   * x[iRow+1 ];
                    yVal -= A.north(iRow)  * x[iRow+Nx];

                    yVal /= A.center(iRow);
                }

                y.swap( x );
            }
        }

        void reviseVelocity() const
        {
        }
};

BORA_NAMESPACE_END

#endif

