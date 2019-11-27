//--------------//
// EnrightTest.h //
//-------------------------------------------------------//
// author: Julie Jang @ Dexter Studios                   //
// last update: 2018.04.10                               //
//-------------------------------------------------------//

#ifndef _BoraEnrightTest_h_
#define _BoraEnrightTest_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

class EnrightTest
{
	private:

		Grid             _grd;
		VectorDenseField _vel;
		ScalarDenseField _lvs;
		Advector         _adv;

		Surfacer		 _surfacer;
		TriangleMesh	 _outputTriMesh;

	public:

		EnrightTest();

		void initialize( int gridRes );
		void initialize( int gridRes, std::string& filePath );
		void setAdvectionConditions( float cfl_number, int minSubSteps, int maxSubSteps, float Dt, AdvectionScheme scheme );
		int  update( float t );
		void exportCache( std::string& filePath );
		void drawWireframe();
		void drawVelocityField( bool isDraw2D, float& scale, Vec3f& sliceRatio );
		void drawScalarField();

	private:

		void _initialize( int gridRes );
		void _setVelocityField( float t );
};

BORA_NAMESPACE_END

#endif

