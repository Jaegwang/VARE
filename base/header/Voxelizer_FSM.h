//--------------//
// Voxelizer_FSM.h //
//-------------------------------------------------------//
// author: Julie Jang @ Dexter Studios                   //
// last update: 2018.04.10                               //
//-------------------------------------------------------//

#ifndef _BoraVoxelizer_FSM_h_
#define _BoraVoxelizer_FSM_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

class Voxelizer_FSM
{
	private:

		bool _initialState;
		bool _entireDomain;
		int _iMax, _jMax, _kMax; // maximum index
		int _I0, _I1, _J0, _J1, _K0, _K1; // computing index range

		float _eps; //epsilon of voxel space (?)
		MarkerDenseField* _stt; // closest triangle

		ScalarDenseField* _lvs; // pointer to signed distance field
		//ZVectorField* _svl; // pointer to solid velocity field

		Vec3fArray*  _vPos;
		//ZVectorArray* _vVel;
		IndexArray*   _triangles;

		float _dx, _dy, _dz;

		const float BORA_LARGE = 1e+30f;

	public:

		Voxelizer_FSM();

		void reset();

		// lvs: signed distance field
		// svl: mesh displacement field
		// mesh: in voxel space   //not world?
		// vDsp: vertex displacements in voxel space
		void addMesh( ScalarDenseField& lvs, MarkerDenseField& stt, TriangleMesh& mesh, bool entireDomain );
		//void addMesh( ZScalarField& lvs, ZVectorField& svl, ZTriMesh& mesh, ZVectorArray& vVel, bool entireDomain );

	private:

		void compute();

		void _tagInterfacialElements();
		void sweep( int di, int dj, int dk );
 		void update( int i0, int j0, int k0 );

		int _baryCoords( Vec3f& P, const Vec3f& A, const Vec3f& B, const Vec3f& C, int whichPlane, Vec3f& baryCoords );
		bool _isInsideTriangle( const Vec3f& baryCoords );
		Vec3f _normal( const Vec3f& A, const Vec3f& B, const Vec3f& C );
		bool _hasPhi( int state );
		float _solvePhi( float p, float q, float r );
		bool _updatePhi( float& phi, float candidate );
};

class FSMState
{
	public:

		enum _FSMState
		{
			far       = 0,
			interface = 1,
			updated   = 2,
		};

	public:

		FSMState() {}
};

/**
	Print the information of the bounding box.
	@param[in] os The output stream object.
	@param[in] solver The solver to print.
	@return The output stream object.
*/
std::ostream& operator<<( std::ostream& os, const Voxelizer_FSM& solver );

BORA_NAMESPACE_END

#endif

