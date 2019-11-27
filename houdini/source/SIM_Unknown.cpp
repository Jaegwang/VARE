//-----------------//
// SIM_Unknown.cpp //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.12.20                               //
//-------------------------------------------------------//

#include <SIM_Unknown.h>

SIM_Unknown::SIM_Unknown( const SIM_DataFactory* factory ) : BaseClass( factory )
{
}

SIM_Unknown::~SIM_Unknown()
{
}

const SIM_DopDescription* SIM_Unknown::getDescription()
{
    static PRM_Template nodeParameters[] =
    {
        houIntParameter( "startframe" , "Start Frame", 1 ),
        PRM_Template()
    };

    static SIM_DopDescription theDopDescription(
        true,
        "bora_doptest",
        "Gas Test",
        "Solver",
        classname(),
        nodeParameters);

    setGasDescription( theDopDescription );

    return &theDopDescription;
}

bool SIM_Unknown::solveGasSubclass( SIM_Engine& engine, SIM_Object* obj, SIM_Time time, SIM_Time timestep )
{
    SIM_GeometryCopy* geo = 0;

    geo = getGeometryCopy( obj, GAS_NAME_GEOMETRY );
    if( !geo )
    {
        // error Message.
    }

    SIM_GeometryAutoWriteLock lock( geo );
    GU_Detail& gdp = lock.getGdp();

    std::cout<< "compute" << std::endl;

    return true;
}


