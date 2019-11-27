//-----------------//
// OceanParams.cpp //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.09.05                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

OceanParams::OceanParams()
{
    // nothing to do
}

OceanParams& OceanParams::operator=( const OceanParams& params )
{
    #define CopyOceanParam(name) \
    {                            \
        name = params.name;      \
    }

    CopyOceanParam( dispersionRelationshipType );
    CopyOceanParam( spectrumType               );
    CopyOceanParam( directionalSpreadingType   );
    CopyOceanParam( filterType                 );
    CopyOceanParam( randomType                 );
    CopyOceanParam( numThreads                 );
    CopyOceanParam( grainSize                  );
    CopyOceanParam( gridLevel                  );
    CopyOceanParam( physicalLength             );
    CopyOceanParam( sceneConvertingScale       );
    CopyOceanParam( gravity                    );
    CopyOceanParam( surfaceTension             );
    CopyOceanParam( density                    );
    CopyOceanParam( depth                      );
    CopyOceanParam( windDirection              );
    CopyOceanParam( windSpeed                  );
    CopyOceanParam( flowSpeed                  );
    CopyOceanParam( fetch                      );
    CopyOceanParam( swell                      );
    CopyOceanParam( filterSoftWidth            );
    CopyOceanParam( filterSmallWavelength      );
    CopyOceanParam( filterBigWavelength        );
    CopyOceanParam( randomSeed                 );
    CopyOceanParam( crestGain                  );
    CopyOceanParam( crestBias                  );
    CopyOceanParam( crestAccumulation          );
    CopyOceanParam( crestDecay                 );
    CopyOceanParam( timeOffset                 );
    CopyOceanParam( timeScale                  );
    CopyOceanParam( loopingDuration            );
    CopyOceanParam( amplitudeGain              );
    CopyOceanParam( pinch                      );

    #undef CopyOceanParam

    return (*this);
}

bool OceanParams::toReInit( const OceanParams& params ) const
{
    bool different = false;

    #define CheckOceanParam(name) \
    {                             \
        if( name != params.name ) \
        {                         \
            different = true;     \
        }                         \
    }

    CheckOceanParam( dispersionRelationshipType );
    CheckOceanParam( spectrumType               );
    CheckOceanParam( directionalSpreadingType   );
    CheckOceanParam( filterType                 );
    CheckOceanParam( randomType                 );
    CheckOceanParam( numThreads                 );
    CheckOceanParam( grainSize                  );
    CheckOceanParam( gridLevel                  );
    CheckOceanParam( physicalLength             );
    CheckOceanParam( sceneConvertingScale       );
    CheckOceanParam( gravity                    );
    CheckOceanParam( surfaceTension             );
    CheckOceanParam( density                    );
    CheckOceanParam( depth                      );
    CheckOceanParam( windDirection              );
    CheckOceanParam( windSpeed                  );
    CheckOceanParam( flowSpeed                  );
    CheckOceanParam( fetch                      );
    CheckOceanParam( swell                      );
    CheckOceanParam( filterSoftWidth            );
    CheckOceanParam( filterSmallWavelength      );
    CheckOceanParam( filterBigWavelength        );
    CheckOceanParam( randomSeed                 );
    CheckOceanParam( crestGain                  );
    CheckOceanParam( crestBias                  );
    CheckOceanParam( crestAccumulation          );
    CheckOceanParam( crestDecay                 );
    CheckOceanParam( timeOffset                 );
    CheckOceanParam( timeScale                  );
    CheckOceanParam( loopingDuration            );
    CheckOceanParam( amplitudeGain              );
    CheckOceanParam( pinch                      );

    #undef CheckOceanParam

    return ( different ? true : false );
}

int OceanParams::resolution() const
{
    return ( 1 << gridLevel ); // = 2^gridLevel
}

float OceanParams::geometricLength() const
{
    return ( physicalLength * sceneConvertingScale );
}

bool OceanParams::looopable() const
{
    if( crestAccumulation ) { return false; }
    if( Abs(flowSpeed) > 0.000001f ) { return false; }
    return true;
}

bool OceanParams::save( const char* filePath, const char* fileName ) const
{
    std::string filePathNameStr;
    {
        filePathNameStr += filePath;
        filePathNameStr += "/";
        filePathNameStr += fileName;
        filePathNameStr += ".oceanParams";
    }

    JSONString js;

    js.append( "fileName", fileName );

    int type = 0;

    type = (int)dispersionRelationshipType;
    js.append( "dispersionRelationshipType", type );

    type = (int)spectrumType;
    js.append( "spectrumType", type );

    type = (int)directionalSpreadingType;
    js.append( "directionalSpreadingType", type );

    type = (int)filterType;
    js.append( "filterType", type );

    type = (int)randomType;
    js.append( "randomType", type );

    js.append( "numThreads",            numThreads            );
    js.append( "grainSize",             grainSize             );
    js.append( "gridLevel",             gridLevel             );
    js.append( "physicalLength",        physicalLength        );
    js.append( "sceneConvertingScale",  sceneConvertingScale  );
    js.append( "gravity",               gravity               );
    js.append( "surfaceTension",        surfaceTension        );
    js.append( "density",               density               );
    js.append( "depth",                 depth                 );
    js.append( "windDirection",         windDirection         );
    js.append( "windSpeed",             windSpeed             );
    js.append( "flowSpeed",             flowSpeed             );
    js.append( "fetch",                 fetch                 );
    js.append( "swell",                 swell                 );
    js.append( "filterSoftWidth",       filterSoftWidth       );
    js.append( "filterSmallWavelength", filterSmallWavelength );
    js.append( "filterBigWavelength",   filterBigWavelength   );
    js.append( "randomSeed",            randomSeed            );
    js.append( "crestGain",             crestGain             );
    js.append( "crestBias",             crestBias             );
    js.append( "crestAccumulation",     crestAccumulation     );
    js.append( "crestDecay",            crestDecay            );
    js.append( "timeOffset",            timeOffset            );
    js.append( "timeScale",             timeScale             );
    js.append( "loopingDuration",       loopingDuration       );
    js.append( "amplitudeGain",         amplitudeGain         );
    js.append( "pinch",                 pinch                 );

    return js.save( filePathNameStr.c_str() );
}

bool OceanParams::load( const char* filePath, const char* fileName )
{
    std::string filePathNameStr;
    {
        filePathNameStr += filePath;
        filePathNameStr += "/";
        filePathNameStr += fileName;
        filePathNameStr += ".oceanParams";
    }

    JSONString js;

    if( js.load( filePathNameStr.c_str() ) == false )
    {
        return false;
    }

    int type = 0;

    js.get( "dispersionRelationshipType", type );
    dispersionRelationshipType = (OceanDispersionRelationshipType)type;

    js.get( "spectrumType", type );
    spectrumType = (OceanSpectrumType)type;

    js.get( "directionalSpreadingType", type );
    directionalSpreadingType = (OceanDirectionalSpreadingType)type;

    js.get( "filterType", type );
    filterType = (OceanFilterType)type;

    js.get( "randomType", type );
    randomType = (OceanRandomType)type;

    js.get( "numThreads",            numThreads            );
    js.get( "grainSize",             grainSize             );
    js.get( "gridLevel",             gridLevel             );
    js.get( "physicalLength",        physicalLength        );
    js.get( "sceneConvertingScale",  sceneConvertingScale  );
    js.get( "gravity",               gravity               );
    js.get( "surfaceTension",        surfaceTension        );
    js.get( "density",               density               );
    js.get( "depth",                 depth                 );
    js.get( "windDirection",         windDirection         );
    js.get( "windSpeed",             windSpeed             );
    js.get( "flowSpeed",             flowSpeed             );
    js.get( "fetch",                 fetch                 );
    js.get( "swell",                 swell                 );
    js.get( "filterSoftWidth",       filterSoftWidth       );
    js.get( "filterSmallWavelength", filterSmallWavelength );
    js.get( "filterBigWavelength",   filterBigWavelength   );
    js.get( "randomSeed",            randomSeed            );
    js.get( "crestGain",             crestGain             );
    js.get( "crestBias",             crestBias             );
    js.get( "crestAccumulation",     crestAccumulation     );
    js.get( "crestDecay",            crestDecay            );
    js.get( "timeOffset",            timeOffset            );
    js.get( "timeScale",             timeScale             );
    js.get( "loopingDuration",       loopingDuration       );
    js.get( "amplitudeGain",         amplitudeGain         );
    js.get( "pinch",                 pinch                 );

    return true;
}

std::ostream& operator<<( std::ostream& os, const OceanParams& p )
{
    os << "OceanParams" << ENDL;

    os << "dispersionRelationshipType: " << p.dispersionRelationshipType << ENDL;
    os << "spectrumType              : " << p.spectrumType               << ENDL;
    os << "directionalSpreadingType  : " << p.directionalSpreadingType   << ENDL;
    os << "filterType                : " << p.filterType                 << ENDL;
    os << "randomType                : " << p.randomType                 << ENDL;

    os << "numThreads                : " << p.numThreads                 << ENDL;
    os << "grainSize                 : " << p.grainSize                  << ENDL;
    os << "gridLevel                 : " << p.gridLevel                  << ENDL;
    os << "physicalLength            : " << p.physicalLength             << ENDL;
    os << "sceneConvertingScale      : " << p.sceneConvertingScale       << ENDL;
    os << "gravity                   : " << p.gravity                    << ENDL;
    os << "surfaceTension            : " << p.surfaceTension             << ENDL;
    os << "density                   : " << p.density                    << ENDL;
    os << "depth                     : " << p.depth                      << ENDL;
    os << "windDirection             : " << p.windDirection              << ENDL;
    os << "windSpeed                 : " << p.windSpeed                  << ENDL;
    os << "flowSpeed                 : " << p.flowSpeed                  << ENDL;
    os << "fetch                     : " << p.fetch                      << ENDL;
    os << "swell                     : " << p.swell                      << ENDL;
    os << "filterSoftWidth           : " << p.filterSoftWidth            << ENDL;
    os << "filterSmallWavelength     : " << p.filterSmallWavelength      << ENDL;
    os << "filterBigWavelength       : " << p.filterBigWavelength        << ENDL;
    os << "randomSeed                : " << p.randomSeed                 << ENDL;
    os << "crestGain                 : " << p.crestGain                  << ENDL;
    os << "crestBias                 : " << p.crestBias                  << ENDL;
    os << "crestAccumulation         : " << p.crestAccumulation          << ENDL;
    os << "crestDecay                : " << p.crestDecay                 << ENDL;

    os << "timeOffset                : " << p.timeOffset                 << ENDL;
    os << "timeScale                 : " << p.timeScale                  << ENDL;
    os << "loopingDuration           : " << p.loopingDuration            << ENDL;
    os << "amplitudeGain             : " << p.amplitudeGain              << ENDL;
    os << "pinch                     : " << p.pinch                      << ENDL;

	return os;
}

BORA_NAMESPACE_END

