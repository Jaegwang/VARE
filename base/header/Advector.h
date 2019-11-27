//------------//
// Advector.h //
//-------------------------------------------------------//
// author: Julie Jang @ Dexter Studios                   //
// last update: 2018.04.25                               //
//-------------------------------------------------------//

#ifndef _BoraAdvector_h_
#define _BoraAdvector_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

class Advector
{
    private:

        size_t Nx = 0;
        size_t Ny = 0;
        size_t Nz = 0;

        ScalarDenseField _tmpScalarField;
        VectorDenseField _tmpVectorField;

		float			 _cfl = 1.f;
		int				 _minSubSteps = 1;
		int				 _maxSubSteps = 2;
		float			 _Dt = 1.f;
		AdvectionScheme  _advScheme = kLinear;

    public:

        Advector();

		void set( float cfl_number, int minSubSteps, int maxSubSteps, float Dt, AdvectionScheme scheme=kLinear );

        int advect( ScalarDenseField& s, const VectorDenseField& v );
        int advect( VectorDenseField& s, const VectorDenseField& v );

//    private:

        void byLinear( ScalarDenseField& s, const VectorDenseField& v, int substeps, float dt );
        void byLinear( VectorDenseField& s, const VectorDenseField& v, int substeps, float dt );

        void byRK2( ScalarDenseField& s, const VectorDenseField& v, int substeps, float dt );
        void byRK2( VectorDenseField& s, const VectorDenseField& v, int substeps, float dt );

        void byRK3( ScalarDenseField& s, const VectorDenseField& v, int substeps, float dt );
        void byRK3( VectorDenseField& s, const VectorDenseField& v, int substeps, float dt );

        void byRK4( ScalarDenseField& s, const VectorDenseField& v, int substeps, float dt );
        void byRK4( VectorDenseField& s, const VectorDenseField& v, int substeps, float dt );

        void byMacCormack( ScalarDenseField& s, const VectorDenseField& v, int substeps, float dt );
        void byMacCormack( VectorDenseField& s, const VectorDenseField& v, int substeps, float dt );

        void byBFECC( ScalarDenseField& s, const VectorDenseField& v, int substeps, float dt );
        void byBFECC( VectorDenseField& s, const VectorDenseField& v, int substeps, float dt );
};

BORA_NAMESPACE_END

#endif

