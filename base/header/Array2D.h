//-----------//
// Array2D.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.03.29                               //
//-------------------------------------------------------//

#ifndef _BoraArray2D_h_
#define _BoraArray2D_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

template<class TT>
class Array2D : public Array<TT>
{
    protected:

        size_t _nx = 0;
        size_t _ny = 0;

    public:

        BORA_FUNC_QUAL
        Array2D();

        BORA_FUNC_QUAL
        Array2D( const Array2D<TT>& a );

        Array2D( const size_t Nx, const size_t Ny, const MemorySpace memorySpace=kHost );

        BORA_FUNC_QUAL
        virtual ~Array2D();

        void initialize( const size_t Nx, const size_t Ny, const MemorySpace memorySpace=kHost );

        void finalize();

        void clear();

        void resize( const size_t Nx, const size_t Ny );

        Array2D<TT>& operator=( const Array2D<TT>& a );

        bool copyFrom( const Array2D<TT>& a );
        bool copyTo( Array2D<TT>& a ) const;

		void write( std::ofstream& fout ) const;
		void read ( std::ifstream& fin );

		bool save( const char* filePathName ) const;
        bool load( const char* filePathName );

        /////////////////////////////
        // inline member functions //

        BORA_FUNC_QUAL size_t nx() const { return _nx; }
        BORA_FUNC_QUAL size_t ny() const { return _ny; }

        BORA_FUNC_QUAL size_t stride() const { return _nx; }

        BORA_FUNC_QUAL TT& operator()( const size_t& i, const size_t& j )
        {
            return Array<TT>::_data[i+j*_nx];
        }

        BORA_FUNC_QUAL const TT& operator()( const size_t& i, const size_t& j ) const
        {
            return Array<TT>::_data[i+j*_nx];
        }
};

////////////////////////////////////
// member function implementation //

template <class TT>
BORA_FUNC_QUAL
Array2D<TT>::Array2D()
{
    // nothing to do
}

template <class TT>
BORA_FUNC_QUAL
Array2D<TT>::Array2D( const Array2D& a )
: Array<TT>::Array()
{
    Array2D::operator=( a );
}

template <class TT>
Array2D<TT>::Array2D( const size_t Nx, const size_t Ny, const MemorySpace memorySpace )
{
    Array2D::initialize( Nx, Ny, memorySpace );
}

template <class TT>
BORA_FUNC_QUAL
Array2D<TT>::~Array2D()
{
    Array<TT>::finalize();
}

template <class TT>
void Array2D<TT>::initialize( const size_t Nx, const size_t Ny, const MemorySpace memorySpace )
{
    Array<TT>::initialize( Nx*Ny, memorySpace );

    _nx = Nx;
    _ny = Ny;
}

template <class TT>
void Array2D<TT>::finalize()
{
    Array<TT>::finalize();

    _nx = _ny = 0;
}

template <class TT>
void Array2D<TT>::clear()
{
    Array<TT>::_size = 0;

    _nx = _ny = 0;
}

template <class TT>
void Array2D<TT>::resize( const size_t Nx, const size_t Ny )
{
    if( ( _nx == Nx ) && ( _ny == Ny ) ) { return; }

    Array<TT>::resize( Nx*Ny );

    _nx = Nx;
    _ny = Ny;
}

template <class TT>
Array2D<TT>& Array2D<TT>::operator=( const Array2D<TT>& a )
{
    Array<TT>::operator=( a );

    _nx = a._nx;
    _ny = a._ny;
}

template <class TT>
bool Array2D<TT>::copyFrom( const Array2D<TT>& a )
{
    if( Array<TT>::copyFrom( a ) )
    {
        _nx = a._nx;
        _ny = a._ny;

        return true;
    }
    else
    {
        _nx = _ny = 0;

        return false;
    }
}

template <class TT>
bool Array2D<TT>::copyTo( Array2D<TT>& a ) const
{
    if( Array<TT>::copyTo( a ) )
    {
        a._nx = _nx;
        a._ny = _ny;

        return true;
    }
    else
    {
        a._nx = a._ny = 0;

        return false;
    }
}

template <class TT>
void Array2D<TT>::write( std::ofstream& fout ) const
{
    if( Array<TT>::_memorySpace == kDevice )
    {
        COUT << "Error@Array2D::write(): Not supported for the device array." << ENDL;
        return;
    }

    fout.write( (char*)&_nx, sizeof(size_t) );
    fout.write( (char*)&_ny, sizeof(size_t) );

    if( _nx*_ny > 0 )
    {
        fout.write( (char*)Array<TT>::_data, sizeof(TT)*_nx*_ny );
    }
}

template <class TT>
void Array2D<TT>::read( std::ifstream& fin )
{
    if( Array<TT>::_memorySpace == kDevice )
    {
        COUT << "Error@Array2D::read(): Not supported for the device array." << ENDL;
        return;
    }

    int Nx = 0;
    fin.read( (char*)&Nx, sizeof(size_t) );

    int Ny = 0;
    fin.read( (char*)&Ny, sizeof(size_t) );

    if( Nx*Ny > 0 )
    {
        Array2D::resize( Nx, Ny );
        fin.read( (char*)Array<TT>::_data, sizeof(TT)*Nx*Ny );
    }
    else
    {
        Array2D::finalize();
    }
}

template <class TT>
bool Array2D<TT>::save( const char* filePathName ) const
{
    if( Array<TT>::_memorySpace == kDevice )
    {
        COUT << "Error@Array2D::save(): Not supported for the device array." << ENDL;
        return false;
    }

    std::ofstream fout( filePathName, std::ios::out|std::ios::binary|std::ios::trunc );

    if( fout.fail() || !fout.is_open() )
    {
        COUT << "Error@Array2D::save(): Failed to save file: " << filePathName << ENDL;
        return false;
    }

    Array2D::write( fout );

    fout.close();

    return true;
}

template <class TT>
bool Array2D<TT>::load( const char* filePathName )
{
    Array2D::clear();
    
    if( Array<TT>::_memorySpace == kDevice )
    {
        COUT << "Error@Array2D::load(): Not supported for the device array." << ENDL;
        return false;
    }

    std::ifstream fin( filePathName, std::ios::in|std::ios::binary );

    if( fin.fail() )
    {
        COUT << "Error@Array2D::load(): Failed to load file." << ENDL;
        return false;
    }

    Array2D::read( fin );

    fin.close();

    return true;
}

BORA_NAMESPACE_END

#endif

