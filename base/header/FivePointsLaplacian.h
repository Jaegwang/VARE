//-----------------------//
// FivePointsLaplacian.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
//         Julie Jang @ Dexter Studios                   //
// last update: 2018.05.15                               //
//-------------------------------------------------------//

#ifndef _BoraFivePointsLaplacian_h_
#define _BoraFivePointsLaplacian_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

template <typename T>
class FivePointsLaplacian
{
    private:

        size_t _nx=0, _ny=0;

        // frequently asked variables
        size_t _nxny = 0;

        Array<T> a;

    public:

        FivePointsLaplacian()
        {
            // nothing to do
        }

        // set
        BORA_FUNC_QUAL T& south ( const size_t& iRow ) { return a[5*iRow  ]; }
        BORA_FUNC_QUAL T& west  ( const size_t& iRow ) { return a[5*iRow+1]; }
        BORA_FUNC_QUAL T& center( const size_t& iRow ) { return a[5*iRow+2]; }
        BORA_FUNC_QUAL T& east  ( const size_t& iRow ) { return a[5*iRow+3]; }
        BORA_FUNC_QUAL T& north ( const size_t& iRow ) { return a[5*iRow+4]; }

        // get
        BORA_FUNC_QUAL const T& south ( const size_t& iRow ) const { return a[5*iRow  ]; }
        BORA_FUNC_QUAL const T& west  ( const size_t& iRow ) const { return a[5*iRow+1]; }
        BORA_FUNC_QUAL const T& center( const size_t& iRow ) const { return a[5*iRow+2]; }
        BORA_FUNC_QUAL const T& east  ( const size_t& iRow ) const { return a[5*iRow+3]; }
        BORA_FUNC_QUAL const T& north ( const size_t& iRow ) const { return a[5*iRow+4]; }

        BORA_FUNC_QUAL size_t nx() const { return _nx; }
        BORA_FUNC_QUAL size_t ny() const { return _ny; }

        BORA_FUNC_QUAL size_t dimension() const { return (_nx*_ny); }

		BORA_FUNC_QUAL void getIndices( const size_t& idx, size_t& i, size_t& j ) const
		{
			i = idx / _nx;
			j = idx % _nx;
		}

        void build( const Grid2D& grid )
        {
			a.initialize( grid.numCells()*5, kUnified );
            a.zeroize();

			_nx = grid.nx();
			_ny = grid.ny();

            _nxny = _nx * _ny;

            const size_t N = FivePointsLaplacian::dimension();

            const size_t iEnd = _nx-1;
            const size_t jEnd = _ny-1;

            for( size_t j=0; j<_ny; ++j )
            {
                for( size_t i=0; i<_nx; ++i )
                {
                    T& sum = FivePointsLaplacian::center(i);

                         if( j != 0    ) { FivePointsLaplacian::south(i) = -1; ++sum; }
                    else if( j != jEnd ) { FivePointsLaplacian::north(i) = -1; ++sum; }

                         if( i > 0     ) { FivePointsLaplacian::west(i) = -1; ++sum; }
                    else if( i < iEnd  ) { FivePointsLaplacian::east(i) = -1; ++sum; }
                }
            }
        }

//
//        // TODO: boundary conditions
//        void set( const ScalarDenseField2D& boundaryField )
//        {
//			a.initialize( boundaryField.numCells()*5, kUnified );
//            //a.resize( boundaryField.numCells() );
//            a.zeroize();
//
//            const size_t N = FivePointsLaplacian::dimension();
//
//			size_t i,j;
//            #pragma omp parallel for
//            for( size_t idx=0; idx<N; ++idx )
//            {
//
//				getIndices( idx, i, j );
//                T& sum = FivePointsLaplacian::center(idx) = 0;
//
//                if( boundaryField(i,j-1) > 0 ) { FivePointsLaplacian::south(idx) = -1; ++sum; }
//                if( boundaryField(i-1,j) > 0 ) { FivePointsLaplacian::west (idx) = -1; ++sum; }
//                if( boundaryField(i+1,j) > 0 ) { FivePointsLaplacian::east (idx) = -1; ++sum; }
//                if( boundaryField(i,j+1) > 0 ) { FivePointsLaplacian::east (idx) = -1; ++sum; }
//            }
//        }

        // iRow : 0 ~ ( _nx * _ny )
        T dot( const size_t& iRow, const Array<T>& vector ) const
        {
            T ret = 0;

            size_t index = iRow*5;

            if( iRow >= _nx         ) { ret += a[  index] * vector[iRow-_nx]; }
            if( iRow >=  1          ) { ret += a[++index] * vector[iRow- 1 ]; }
                                      { ret += a[++index] * vector[iRow    ]; }
            if( iRow <= _nxny-2     ) { ret += a[++index] * vector[iRow+ 1 ]; }
            if( iRow <= _nxny-_nx-1 ) { ret += a[++index] * vector[iRow+_nx]; }

            return ret;
        }

        // matrix-vector muliplication (Ax=b)
        bool apply( const Array<T>& x, Array<T>& b ) const
        {
            const size_t N = FivePointsLaplacian::dimension();

            if( ( N != x.size() ) || ( N != b.size() ) )
            {
                COUT << "Error@FivePointsLaplacian::apply(): Invalid dimension." << ENDL;
                return false;
            }

            //#pragma omp parallel for
            for( size_t iRow=0; iRow<_nxny; ++iRow )
            {
                b[iRow] = FivePointsLaplacian::dot( iRow, x );
            }

            return true;
        }
};

BORA_NAMESPACE_END

#endif

