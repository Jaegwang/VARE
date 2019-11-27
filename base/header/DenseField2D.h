//----------------//
// DenseField2D.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
//         Jaegwang Lim @ Dexter Studios                 //
//         Julie Jang @ Dexter Studios                   //
// last update: 2018.04.25                               //
//-------------------------------------------------------//

#ifndef _BoraDenseField2D_h_
#define _BoraDenseField2D_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

template <class T>
class DenseField2D : public Grid2D, public Array<T>
{
    public:

        DenseField2D( const MemorySpace memorySpace=kHost )
        {
            DenseField2D::initialize( Grid2D(), memorySpace );
        }

        void initialize( const Grid2D& grid, MemorySpace memorySpace=kHost )
        {
            Grid2D::operator=( grid );
            Array<T>::initialize( grid.numCells(), memorySpace );
        }        

        void setGrid( const Grid2D& grid )
        {
            Grid2D::operator=( grid );
            Array<T>::resize( grid.numCells() );
        }

        BORA_FUNC_QUAL
        T& operator()( const size_t i, const size_t j )
        {
            const size_t index = Grid2D::cellIndex( i, j );
            return Array<T>::operator[]( index );
        }

        BORA_FUNC_QUAL
        const T& operator()( const size_t i, const size_t j ) const
        {
            const size_t index = Grid2D::cellIndex( i, j );
            return Array<T>::operator[]( index );
        }

        BORA_FUNC_QUAL
        T lerp( const Vec2f& p /*voxel space*/ ) const
        {
            const size_t i = Clamp( size_t(p.x-0.5f), (size_t)0, _nx-2 );
            const size_t j = Clamp( size_t(p.y-0.5f), (size_t)0, _ny-2 );

            const Vec2f corner = Grid2D::cellCenter( i, j );

            const float x = Clamp( (p.x - corner.x), 0.f, 1.f );
            const float y = Clamp( (p.y - corner.y), 0.f, 1.f );

            const DenseField2D<T>& f = (*this);

            const T c00 = f(i, j   )*(1.f-x) + f(i+1, j   )*x;
            const T c10 = f(i, j+1 )*(1.f-x) + f(i+1, j+1 )*x;

            return c00*(1.f-y) + c10*y;
        }

        BORA_FUNC_QUAL
        T catrom( const Vec3f& p /*voxel space*/ ) const
        {
            return T(0);
        }

        DenseField2D<T>& operator=( const DenseField2D<T>& field )
        {
            Grid2D::operator=( field );
            Array<T>::operator=( field );

            return (*this);
        }

        bool swap( DenseField2D<T>& field )
        {
            if( Grid2D::operator!=( field ) )
            {
                COUT << "Error@DenseField2D::exchange(): Grid2D mismatch." << ENDL;
                return false;
            }

            Array<T>::swap( field );

            return true;
        }
};

BORA_NAMESPACE_END

#endif

