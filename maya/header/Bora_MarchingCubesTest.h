//--------------------//
// Bora_MarchingCubesTest.h //
//-------------------------------------------------------//
// author: Julie Jang @ Dexter Studios                   //
// last update: 2018.04.10                               //
//-------------------------------------------------------//

#ifndef _Bora_MarchingCubesTest_h_
#define _Bora_MarchingCubesTest_h_
#define STRINGIFY(s) #s

#include <Bora.h>
#include <MayaCommon.h>
#include <MayaUtils.h>
#include <Bora_ViewportDraw.h>

class Bora_MarchingCubesTest : public MPxSurfaceShape
{
    private: // maya

        MObject           nodeObj;
        MString           nodeName;
        MFnDependencyNode nodeFn;
        MFnDagNode        dagNodeFn;
        MDataBlock*       blockPtr;
        bool              isThe1stDraw = true;
		bool              toUpdateAE = false;
		float			  BORA_LARGE = 1e+30f;

    private: // bora


		// level set variables
		TriangleMesh			_triMesh;
		TriangleMesh			_outputTriMesh;
		Surfacer				_marchingCubes;
		ScalarDenseField		_lvs;
		MarkerDenseField		_stt;
		Voxelizer_FMM			_lvsSolver;


    public:

        static MTypeId id;
        static MString name;
        static MString drawDbClassification;
        static MString drawRegistrantId;

        static MObject inputObj;
        static MObject inXFormObj;
        static MObject outputObj;

        static MObject subdivisionObj;
		static MObject inMeshObj;


    public:

        Bora_MarchingCubesTest();
        virtual ~Bora_MarchingCubesTest();

        static void* creator();
        virtual void postConstructor();
        static MStatus initialize();
        virtual MStatus compute( const MPlug&, MDataBlock& );
        virtual bool isBounded() const;
        virtual MBoundingBox boundingBox() const;
        void draw( int );

    private:

        void _autoConnect();

};

class Bora_MarchingCubesTestDrawOverride : public Bora_DrawOverride<Bora_MarchingCubesTest>
{
    public:

        Bora_MarchingCubesTestDrawOverride( const MObject& dgNodeObj )
        : Bora_DrawOverride<Bora_MarchingCubesTest>( dgNodeObj )
        {}
};

#endif

