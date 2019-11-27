//--------------//
// Voxelizer_FMM.h //
//-------------------------------------------------------//
// author: Julie Janf @ Dexter Studios                   //
// last update: 2018.04.10                               //
//-------------------------------------------------------//

#ifndef _BoraVoxelizer_FMM_h_
#define _BoraVoxelizer_FMM_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

class Voxelizer_FMM
{
	private:

		bool _initialState;
		bool _onCell;						// definedd at cell or node

		int   _iMax, _jMax, _kMax;			// max. possible indices
		float _h;							// cell size
		float _eps;							// epsilon of voxel space grid
		float _negRange, _posRange;			// for narrow band fast marching method

		typedef HeapNode<Vec3i,float> HEAPNODE;
		MinHeap<Vec3i,float> _posHeap;
		MaxHeap<Vec3i,float> _negHeap;

		// grid
		Grid			  _grid;
		ScalarDenseField* _lvs;				// pointer to signed distance field
		//ZVectorField3D* _vel;				// pointer to solid velocity field
		MarkerDenseField* _stt;				// state (far, interface, updated, or trial)

		// mesh
		Vec3fArray*   _vPos;				// vertex positions
		//ZVectorArray*  _vVel;				// vertex velocities (displacement)
		IndexArray*    _v012;				// vertex connections //is this the correct datatype?

		const float BORA_LARGE = 1e+30f;


	public:

		Voxelizer_FMM();

		void reset();


		void set( const Grid& grid );
		void set( float h, int maxSubdivision, const AABB3f& bBox );



		// lvs: signed distance field
		// vel: mesh displacement field
		// mesh: in world space
		// vDsp: vertex displacements in world space
		void addMesh( ScalarDenseField& lvs, MarkerDenseField& stt, TriangleMesh& mesh, float negRange, float posRange );

		void finalize();

	private:

		void _tagInterfacialElements();
		void _update( const Vec3i& ijk, int sign );

		Vec3f _closestPointOnTriangle( const Vec3f& P, const Vec3f& A, const Vec3f& B, const Vec3f& C, Vec3f& baryCoords );

		int _baryCoords( Vec3f& P, const Vec3f& A, const Vec3f& B, const Vec3f& C, int whichPlane, Vec3f& baryCoords );

		bool _isInsideTriangle( const Vec3f& baryCoords );
		Vec3f _normal( Vec3f& A, Vec3f& B, Vec3f& C );

		bool _hasPhi( int state );
		float _solvePhi( float p, float q, float r );
		bool _updatePhi( float& phi, float candidate );

};

class FMMState
{
	public:

		enum _FMMState
		{
			none      = 0,
			far       = 1,
			interface = 2,
			updated   = 3,
			trial     = 4
		};

	public:

		FMMState() {}
};

/**
	Print the information of the bounding box.
	@param[in] os The output stream object.
	@param[in] solver The solver to print.
	@return The output stream object.
*/
std::ostream& operator<<( std::ostream& os, const Voxelizer_FMM& solver );

BORA_NAMESPACE_END

#endif

