//-------------//
// OceanTile.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.04.23                               //
//-------------------------------------------------------//

#ifndef _BoraOceanTile_h_
#define _BoraOceanTile_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

class OceanTile
{
    private:

        bool _initialized = false;
        OceanParams _oceanParams;

    private:

        FFT fft; // a wrapper of FFTW

    public:

        // fields in frequency domain
        RealOceanSpectralField    Omega; // angular frequency by dispersion relationship
        ComplexOceanSpectralField H0Pos; // positive-directional height field
        ComplexOceanSpectralField H0Neg; // negative-directional height field
        ComplexOceanSpectralField HSpec; // spectral height field
        ComplexOceanSpectralField TmpSF; // temporary spectral field

        // TmpSF is a spectral field (a field in frequency domain) for temporary use.
        // It is for the input of FFT.

        // fields in spatial domain
        RealOceanSpatialField Dh;   // displacement in height
        RealOceanSpatialField Dx;   // displacement in x (It may seem like a derivative, but it is not.)
        RealOceanSpatialField Dy;   // displacement in y (It may seem like a derivative, but it is not.)
        RealOceanSpatialField J;    // eigenvalues

        // We do not need the first derivatives (Dxx, Dyy, and Dxy) explicitly
        // , because they are just ingredients for making J.
        // We use Dx for Dxx, Dy for Dyy, and J for Dxy temporarily.
        // Dxx = tho(Dx) / tho(x)
        // Dyy = tho(Dy) / tho(y)
        // Dxy = tho(Dx) / tho(y)
        // cf) tho = rounded d = partial derivative symbol = partial

    public:

        OceanTile( const OceanParams& params = OceanParams() );

        bool initialize( const OceanParams& params );

        void update( const float& time );

        const OceanParams& oceanParams() const;

        bool initialized() const;

        float currentTime() const;

    private:

        void allocate();

        void setInitialData();
};

BORA_NAMESPACE_END

#endif

