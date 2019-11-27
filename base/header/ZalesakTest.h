//--------------//
// ZalesakTest.h //
//-------------------------------------------------------//
// author: Julie Jang @ Dexter Studios                   //
// last update: 2018.04.10                               //
//-------------------------------------------------------//

#ifndef _BoraZalesakTest_h_
#define _BoraZalesakTest_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

class ZalesakTest
{
	private:

		Grid			 _grid;
		VectorDenseField _vel;
		ScalarDenseField _lvs;
		Advector		 _adv;

		Surfacer		 _surfacer;
		TriangleMesh	 _outputTriMesh;

	public:

		ZalesakTest();
		void initialize( const float gridRes );
		void setAdvectionConditions( float cfl_number, int minSubSteps, int maxSubSteps, float Dt, AdvectionScheme scheme );
		int  update( const float t );
		void exportCache( std::string& filePath );
		void drawWireframe();
		void drawVelocityField( bool isDraw2D, float& scale, Vec3f& sliceRatio );
		void drawScalarField();


	private:

		void _setVorticityField();

};


BORA_NAMESPACE_END

#endif

