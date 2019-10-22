
#pragma once
#include <VARE.h>

VARE_NAMESPACE_BEGIN

template <class T>
class DenseField : public Grid, public Array<T>
{
    public:

        DenseField( const MemorySpace memorySpace=kHost )
        {
            DenseField::initialize( Grid(), memorySpace );
        }

        void initialize( const Grid& grid, MemorySpace memorySpace=kHost )
        {
            Grid::operator=( grid );
            Array<T>::initialize( grid.numCells(), memorySpace );
        }        

        void setGrid( const Grid& grid )
        {
            Grid::operator=( grid );
            Array<T>::resize( grid.numCells() );
        }

        VARE_UNIFIED
        T& operator()( const size_t i, const size_t j, const size_t k=0 )
        {
            const size_t index = Grid::cellIndex( i, j, k );
            return Array<T>::operator[]( index );
        }

        VARE_UNIFIED
        const T& operator()( const size_t i, const size_t j, const size_t k=0 ) const
        {
            const size_t index = Grid::cellIndex( i, j, k );
            return Array<T>::operator[]( index );
        }

        VARE_UNIFIED
        T lerp( const Vec3f& p /*voxel space*/ ) const
        {
            const size_t i = Clamp( int(p.x-0.5f), 0, (int)_nx-2 );
            const size_t j = Clamp( int(p.y-0.5f), 0, (int)_ny-2 );
            const size_t k = Clamp( int(p.z-0.5f), 0, (int)_nz-2 );

            const Vec3f corner = Grid::cellCenter( i, j, k );

            const float x = Clamp( (p.x - corner.x), 0.f, 1.f );
            const float y = Clamp( (p.y - corner.y), 0.f, 1.f );
            const float z = Clamp( (p.z - corner.z), 0.f, 1.f );

            const DenseField<T>& f = (*this);

            const T c00 = f(i, j,   k  )*(1.f-x) + f(i+1, j,   k  )*x;
            const T c01 = f(i, j,   k+1)*(1.f-x) + f(i+1, j,   k+1)*x;
            const T c10 = f(i, j+1, k  )*(1.f-x) + f(i+1, j+1, k  )*x;
            const T c11 = f(i, j+1, k+1)*(1.f-x) + f(i+1, j+1, k+1)*x;

            const T c0 = c00*(1.f-y) + c10*y;
            const T c1 = c01*(1.f-y) + c11*y;

            return c0*(1.f-z) + c1*z;
        }

        VARE_UNIFIED
        T catrom( const Vec2f& p /*voxel space*/ ) const
        {
            return T(0);
        }

        DenseField<T>& operator=( const DenseField<T>& field )
        {
            Grid::operator=( field );
            Array<T>::operator=( field );

            return (*this);
        }

        bool swap( DenseField<T>& field )
        {
            if( Grid::operator!=( field ) )
            {
                COUT << "Error@DenseField::exchange(): Grid mismatch." << ENDL;
                return false;
            }

            Array<T>::swap( field );

            return true;
        }
};

typedef DenseField<float> ScalarDenseField;
typedef DenseField<float> FloatDenseField;
typedef DenseField<Vec3f> VectorDenseField;
typedef DenseField<Vec3f> Vec3fDenseField;

VARE_NAMESPACE_END

