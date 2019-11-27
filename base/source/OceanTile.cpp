//---------------//
// OceanTile.cpp //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.04.23                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

OceanTile::OceanTile( const OceanParams& params )
{
    OceanTile::initialize( params );
}

bool OceanTile::initialize( const OceanParams& params )
{
    const bool toReInit = _oceanParams.toReInit( params );

    _oceanParams = params;

    if( toReInit )
    {
        OceanTile::allocate();
        OceanTile::setInitialData();

        _initialized = true;
    }

    const bool reInitialized = toReInit;

    return reInitialized;
}

void OceanTile::allocate()
{
    Omega.resize( _oceanParams.gridLevel );
    H0Pos.resize( _oceanParams.gridLevel );
    H0Neg.resize( _oceanParams.gridLevel );
    HSpec.resize( _oceanParams.gridLevel );
    TmpSF.resize( _oceanParams.gridLevel );

    Dh.resize( _oceanParams.gridLevel, 1 );
    Dx.resize( _oceanParams.gridLevel, 1 );
    Dy.resize( _oceanParams.gridLevel, 1 );
    J .resize( _oceanParams.gridLevel, 1 );

    fft.initialize( HSpec, Dh, _oceanParams.numThreads );
}

void OceanTile::setInitialData()
{
    const int    N = _oceanParams.resolution();
    const float& L = _oceanParams.physicalLength;

    const size_t& grainSize = _oceanParams.grainSize;

    OceanFunctor_InitialState F;
    {
        F.filter     .set( _oceanParams );
        F.spectrum   .set( _oceanParams );
        F.random     .set( _oceanParams );
        F.spreading  .set( _oceanParams );
        F.dispersion .set( _oceanParams );

        F.Omega = Omega.pointer();
        F.H0Pos = H0Pos.pointer();
        F.H0Neg = H0Neg.pointer();

        F.windAngle = DegToRad( _oceanParams.windDirection );
        F.PI2_L = PI2 / L;
    }

    ComputeSpectralField<OceanFunctor_InitialState>( F, N, L, grainSize );
}

void OceanTile::update( const float& time )
{
    if( _initialized == false ) { return; }

    const int   N = _oceanParams.resolution();
    const float L = _oceanParams.physicalLength;

    const float timeOffset      = _oceanParams.timeOffset;
    const float timeScale       = _oceanParams.timeScale;
    const float loopingDuration = _oceanParams.loopingDuration;

    const size_t grainSize = _oceanParams.grainSize;

    // H0Pos & H0Neg -> HSpec
    {
        OceanFunctor_Dh F;
        {
            F.time            = time + timeOffset;
            F.timeScale       = timeScale;
            F.loopingDuration = loopingDuration;

            F.Omega = Omega.pointer();
            F.H0Pos = H0Pos.pointer();
            F.H0Neg = H0Neg.pointer();
            F.HSpec = HSpec.pointer();
        }
        ComputeSpectralField<OceanFunctor_Dh>( F, N, L, grainSize );
    }

    // HSpec -> Dxx
    // We use Dx for computing Dxx temporarily.
    {
        OceanFunctor_Dxx F;
        {
            F.HSpec   = HSpec.pointer();
            F.DxxSpec = TmpSF.pointer();
        }
        ComputeSpectralField<OceanFunctor_Dxx>( F, N, L, grainSize );
        fft.execute( TmpSF, Dx ); // TmpSF -> Dxx (here, Dx = Dxx temporarily)
    }

    // HSpec -> Dyy
    // We use Dy for computing Dyy temporarily.
    {
        OceanFunctor_Dyy F;
        {
            F.HSpec   = HSpec.pointer();
            F.DyySpec = TmpSF.pointer();
        }
        ComputeSpectralField<OceanFunctor_Dyy>( F, N, L, grainSize );
        fft.execute( TmpSF, Dy ); // TmpSF -> Dyy (here, Dy = Dyy temporarily)
    }

    // HSpec -> Dxy
    // We use J for computing Dxy temporarily.
    {
        OceanFunctor_Dxy F;
        {
            F.HSpec   = HSpec.pointer();
            F.DxySpec = TmpSF.pointer();
        }
        ComputeSpectralField<OceanFunctor_Dxy>( F, N, L, grainSize );
        fft.execute( TmpSF, J ); // TmpSF -> Dxy (here, J = Dxy temporarily)
    }

    // Dxx, Dyy, Dxy -> J
    {
        OceanFunctor_J F;
        {
            F.pinch = _oceanParams.pinch;
            F.Dxx = Dx.pointer();
            F.Dyy = Dy.pointer();
            F.Dxy = J.pointer();
            F.J   = J.pointer();
        }
        tbb::parallel_for( tbb::blocked_range<int>(0,Dh.size(),grainSize), F );
    }

    // HSpec -> Dx
    {
        OceanFunctor_Dx F;
        {
            F.HSpec  = HSpec.pointer();
            F.DxSpec = TmpSF.pointer();
        }
        ComputeSpectralField<OceanFunctor_Dx>( F, N, L, grainSize );
        fft.execute( TmpSF, Dx ); // TmpSF -> Dx
    }

    // HSpec -> Dy
    {
        OceanFunctor_Dy F;
        {
            F.HSpec  = HSpec.pointer();
            F.DySpec = TmpSF.pointer();
        }
        ComputeSpectralField<OceanFunctor_Dy>( F, N, L, grainSize );
        fft.execute( TmpSF, Dy ); // TmpSF -> Dy
    }

    // HSpec -> H
    // H(=height) must be computed lastly because FFT modifies the input(=HSpec).
    // fftw_execute_dft_c2r() is one of the new-array execute functions in FFTW 3.3.7.
    // It is a side-effect of the c2r algorithms that they are hard to implement efficiently without overwriting the input.
    {
        fft.execute( HSpec, Dh ); // HSpec -> Dh
    }
}

const OceanParams& OceanTile::oceanParams() const
{
    return _oceanParams;
}

bool OceanTile::initialized() const
{
    return _initialized;
}

BORA_NAMESPACE_END

