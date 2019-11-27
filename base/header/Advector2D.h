//--------------//
// Advector2D.h //
//-------------------------------------------------------//
// author: Julie Jang @ Dexter Studios                   //
// last update: 2018.04.25                               //
//-------------------------------------------------------//

#ifndef _BoraAdvector2D_h_
#define _BoraAdvector2D_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

class Advector2D
{
    private:

        size_t Nx = 0;
        size_t Ny = 0;

        ScalarDenseField2D _tmpScalarField;
        VectorDenseField2D _tmpVectorField;

        float              _cfl = 1.f;
        int 			   _minSubSteps = 1;
        int 			   _maxSubSteps = 2;
		float			   _Dt = 1.f;
		AdvectionScheme    _advScheme = kLinear;

    public:

        Advector2D();

		void set( float cfl_number, int minSubSteps, int maxSubSteps, float Dt, AdvectionScheme scheme=kLinear );

        int advect( ScalarDenseField2D& s, const VectorDenseField2D& v );
        int advect( VectorDenseField2D& s, const VectorDenseField2D& v );


//    private:

        void byLinear( ScalarDenseField2D& s, const VectorDenseField2D& v, int substeps, float dt );
        void byLinear( VectorDenseField2D& s, const VectorDenseField2D& v, int substeps, float dt );

        void byRK2( ScalarDenseField2D& s, const VectorDenseField2D& v, int substeps, float dt );
        void byRK2( VectorDenseField2D& s, const VectorDenseField2D& v, int substeps, float dt );

        void byRK3( ScalarDenseField2D& s, const VectorDenseField2D& v, int substeps, float dt );
        void byRK3( VectorDenseField2D& s, const VectorDenseField2D& v, int substeps, float dt );

        void byRK4( ScalarDenseField2D& s, const VectorDenseField2D& v, int substeps, float dt );
        void byRK4( VectorDenseField2D& s, const VectorDenseField2D& v, int substeps, float dt );

        void byMacCormack( ScalarDenseField2D& s, const VectorDenseField2D& v, int substeps, float dt );
        void byMacCormack( VectorDenseField2D& s, const VectorDenseField2D& v, int substeps, float dt );

        void byBFECC( ScalarDenseField2D& s, const VectorDenseField2D& v, int substeps, float dt );
        void byBFECC( VectorDenseField2D& s, const VectorDenseField2D& v, int substeps, float dt );
};

BORA_NAMESPACE_END

#endif

