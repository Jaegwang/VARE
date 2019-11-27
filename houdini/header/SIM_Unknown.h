//-------------//
// SIM_Unknown.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.12.20                               //
//-------------------------------------------------------//

#pragma once

#include <HouCommon.h>
#include <Bora.h>

class SIM_Unknown : public GAS_SubSolver
{
    protected:
        
        explicit SIM_Unknown( const SIM_DataFactory* factory );
        virtual ~SIM_Unknown();

        virtual bool solveGasSubclass( SIM_Engine& engine, SIM_Object* obj, SIM_Time time, SIM_Time timestep );

    private:

        static const SIM_DopDescription* getDescription();

        DECLARE_STANDARD_GETCASTTOTYPE();
        DECLARE_DATAFACTORY( SIM_Unknown, GAS_SubSolver, "Gas test", getDescription() );
};

