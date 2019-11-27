//---------------//
// OceanParams.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.09.05                               //
//-------------------------------------------------------//

#ifndef _BoraOceanParams_h_
#define _BoraOceanParams_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

enum OceanDispersionRelationshipType
{
    kDeepWater,
    kFiniteDepthWater,
    kVerySmallDepthWater
};

enum OceanSpectrumType
{
    kPhillips,
    kPiersonMoskowitz,
    kJONSWAP,
    kTMA
};

enum OceanDirectionalSpreadingType
{
    kMitsuyasu,
    kHasselmann
};

enum OceanFilterType
{
    kNoFilter,
    kSmoothBandPass
};

enum OceanRandomType
{
    kGaussian,
    kLogNormal
};

class OceanParams
{
    public:

        ///////////////////////////////////////
        // initialization related parameters //

        OceanDispersionRelationshipType dispersionRelationshipType = kDeepWater;
        OceanSpectrumType               spectrumType               = kTMA;
        OceanDirectionalSpreadingType   directionalSpreadingType   = kHasselmann;
        OceanFilterType                 filterType                 = kNoFilter;
        OceanRandomType                 randomType                 = kGaussian;

        int   numThreads            = -1;        // # of threads for use (-1: use all available threads)
        int   grainSize             = 512;       // grain size for TBB
        int   gridLevel             = 9;         // grid resolution exponent
        float physicalLength        = 100.f;     // physical length of the ocean tile (m)
        float sceneConvertingScale  = 10.f;      // so that Maya 1 unit = 0.1m (geometricLength = physicalScale * sceneConvertingScale)
        float gravity               = 9.81f;     // gravitational acceleration (m/sec^2)
        float surfaceTension        = 0.074f;    // surface tension coefficient (N/m)
        float density               = 1000.f;    // water density (kg/m^3)
        float depth                 = 100.f;     // water depth (m)
        float windDirection         = 0.f;       // initially positive x-direction (degrees)
        float windSpeed             = 17.f;      // wind speed (m/sec)
        float flowSpeed             = 0.f;       // flow(=translation) speed along the wind direction
        float fetch                 = 300.f;     // fetch (the size of the wind) (km)
        float swell                 = 0.f;       // swell (directional spreading stuff)
        float filterSoftWidth       = 1.f;       // soft width (filter stuff) (m)
        float filterSmallWavelength = 0.f;       // small wave length (filter stuff) (m)
        float filterBigWavelength   = INFINITE;  // big wave length (filter stuff) (m)
        int   randomSeed            = 0;         // random seed (random stuff)
        float crestGain             = 1.f;       // crest (whitecaps) scaling factor
        float crestBias             = 0.f;       // crest (whitecaps) adding factor
        int   crestAccumulation     = 0;         // crest accumulation on/off flag
        float crestDecay            = 0.4f;      // crest decaying factor (only valid when crestAccumulation is on)

        /////////////////////////////////////////
        // per frame update related parameters //

        float timeOffset            = 0.f;       // time offset
        float timeScale             = 1.f;       // time scale
        float loopingDuration       = 240.f;     // looping duration (unit: sec.)
        float amplitudeGain         = 1.f;       // vertical displacement scaling factor
        float pinch                 = 0.75;      // horizontal displacement scaling factor

    public:

        OceanParams();

        OceanParams& operator=( const OceanParams& params );

        bool toReInit( const OceanParams& params ) const;

        int resolution() const;

        float geometricLength() const;

        bool looopable() const;

        bool save( const char* filePath, const char* fileName ) const;

        bool load( const char* filePath, const char* fileName );
};

std::ostream& operator<<( std::ostream& os, const OceanParams& p );

BORA_NAMESPACE_END

#endif

