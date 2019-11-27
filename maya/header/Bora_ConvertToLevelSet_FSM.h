//--------------------//
// Bora_ConvertToLevelSet_FSM.h //
//-------------------------------------------------------//
// author: Julie Jang @ Dexter Studios                   //
// last update: 2017.04.10                               //
//-------------------------------------------------------//

#ifndef _Bora_ConvertToLevelSet_FSM_h_
#define _Bora_ConvertToLevelSet_FSM_h_
#define STRINGIFY(s) #s

#include <Bora.h>
#include <MayaCommon.h>
#include <MayaUtils.h>
#include <Bora_ViewportDraw.h>

class Bora_ConvertToLevelSet_FSM : public MPxSurfaceShape
{
    private: // maya

        MObject           nodeObj;
        MString           nodeName;
        MFnDependencyNode nodeFn;
        MFnDagNode        dagNodeFn;
        MDataBlock*       blockPtr;
        bool              isThe1stDraw = true;
		bool              toUpdateAE = false;

    private: // bora

		TriangleMesh			_triMesh;
		ScalarDenseField		_lvs;
		MarkerDenseField		_stt;
		Voxelizer_FSM			_lvsSolver;


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

        Bora_ConvertToLevelSet_FSM();
        virtual ~Bora_ConvertToLevelSet_FSM();

        static void* creator();
        virtual void postConstructor();
        static MStatus initialize();
        virtual MStatus compute( const MPlug&, MDataBlock& );
        virtual bool isBounded() const;
        virtual MBoundingBox boundingBox() const;
        void draw( int );

		void drawSlice( const Vec3i& whichSlice, const Vec3f& sliceRatio,
                        bool smoothPosArea=true, const Vec4f& farPos=Vec4f(0.f,0.f,1.f,1.f), const Vec4f& nearPos=Vec4f(1.f,1.f,0.f,1.f),
                        bool smoothNegArea=true, const Vec4f& farNeg=Vec4f(1.f,0.f,0.f,1.f), const Vec4f& nearNeg=Vec4f(1.f,1.f,0.f,1.f),
                        float elementSize=1.f ) const;



    private:

        void _autoConnect();

		void _drawXSlice( int i,
                          bool smoothPosArea, const Vec4f& farPos, const Vec4f& nearPos,
                          bool smoothNegArea, const Vec4f& farNeg, const Vec4f& nearNeg,
                          float elementSize ) const;

		void _drawYSlice( int j,
                          bool smoothPosArea, const Vec4f& farPos, const Vec4f& nearPos,
                          bool smoothNegArea, const Vec4f& farNeg, const Vec4f& nearNeg,
                          float elementSize ) const;

		void _drawZSlice( int k,
                          bool smoothPosArea, const Vec4f& farPos, const Vec4f& nearPos,
                          bool smoothNegArea, const Vec4f& farNeg, const Vec4f& nearNeg,
                          float elementSize ) const;

};

class Bora_ConvertToLevelSet_FSMDrawOverride : public Bora_DrawOverride<Bora_ConvertToLevelSet_FSM>
{
    public:

        Bora_ConvertToLevelSet_FSMDrawOverride( const MObject& dgNodeObj )
        : Bora_DrawOverride<Bora_ConvertToLevelSet_FSM>( dgNodeObj )
        {}
};

#endif

