//-----------------//
// EnrightTest2D.h //
//-------------------------------------------------------//
// author: Julie Jang @ Dexter Studios                   //
// last update: 2018.04.25                               //
//-------------------------------------------------------//

#ifndef _BoraEnrightTest2D_h_
#define _BoraEnrightTest2D_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

class EnrightTest2D
{
	private:

		Grid2D             _grd;
		VectorDenseField2D _vel;
		ScalarDenseField2D _lvs;
		Advector2D         _adv;

	public:

		EnrightTest2D();

        void initialize( size_t gridRes );
		void setAdvectionConditions( float cfl_number, int minSubSteps, int maxSubSteps, float Dt, AdvectionScheme scheme );
        int  update( float t );
		void drawVelocityField();
		void drawScalarField();

	private:

		void _setVelocityField( float t );
};

BORA_NAMESPACE_END

#endif

