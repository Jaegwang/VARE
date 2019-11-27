//--------------//
// OceanField.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.03.29                               //
//-------------------------------------------------------//

#ifndef _BoraOceanField_h_
#define _BoraOceanField_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

template <class T>
class OceanSpatialField : public Array2D<T>
{
    private:

        int _pad = 0;

    public:

        OceanSpatialField( const int& gridLevel=0, const int& pad=0 )
        {
            OceanSpatialField::resize( gridLevel, pad );
        }

        void resize( const int& gridLevel, const int& pad=0 )
        {
            _pad = pad;
            Array2D<T>::resize( PowerOfTwo(gridLevel)+pad, PowerOfTwo(gridLevel)+pad );
        }

        int pad() const { return _pad; }

        int nx() const { return ( Array2D<T>::_nx - _pad ); }
        int ny() const { return ( Array2D<T>::_ny - _pad ); }

        int resolution() const { return ( Array2D<T>::_ny - _pad ); }
};

typedef OceanSpatialField<float>    RealOceanSpatialField;
typedef OceanSpatialField<Complexf> ComplexOceanSpatialField;

template <class T>
class OceanSpectralField : public Array2D<T>
{
    public:

        OceanSpectralField( const int& gridLevel=0 )
        {
            OceanSpectralField::resize( gridLevel );
        }

        void resize( const int& gridLevel )
        {
            Array2D<T>::resize( (PowerOfTwo(gridLevel)/2)+1, PowerOfTwo(gridLevel) );
        }
};

typedef OceanSpectralField<float>    RealOceanSpectralField;
typedef OceanSpectralField<Complexf> ComplexOceanSpectralField;

BORA_NAMESPACE_END

#endif

