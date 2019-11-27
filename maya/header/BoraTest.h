//-------------//
// BoraTest.h //
//-------------------------------------------------------//
// author: Julie Jang @ Dexter Studios                   //
// last update: 2018.04.10                               //
//-------------------------------------------------------//

#ifndef _BoraTest_h_
#define _BoraTest_h_

#include <Bora.h>
#include <MayaCommon.h>
#include <MayaUtils.h>
#include <Bora_ViewportDraw.h>

class BoraTest : public MPxLocatorNode
{
	private:

        bool isThe1stTimeDraw = true;
		bool toUpdateAE = false;

		MObject           nodeObj;
		MString           nodeName;
		MFnDagNode        dagNodeFn;
        MDataBlock*       blockPtr;
		MFnDependencyNode nodeFn;

    private:

		EnrightTest		  _enrightTest;
		ZalesakTest		  _zalesakTest;
		short			  _testType;
		short			  _advectionScheme;
		int				  _gridRes;
		std::string		  _inputAlembicPath;

		ScalarDenseField  _lvs;

		TriangleMesh	  _triMeshTest;

	public:

		static MTypeId id;
		static MString name;
		static MString drawDbClassification;
		static MString drawRegistrantId;

    public:

		static MObject inTimeObj;
		static MObject outputObj;
		static MObject testTypeObj;
		static MObject advectionSchemeObj;
		static MObject inputAlembicPathObj;
		static MObject outputAlembicPathObj;
		static MObject gridResObj;
		static MObject scaleObj;
		static MObject xSliceObj;
		static MObject ySliceObj;
		static MObject zSliceObj;
		static MObject inMeshObj;

	public:

		static void* creator();
		virtual void postConstructor();
		static MStatus initialize();
		virtual MStatus compute( const MPlug&, MDataBlock& );
		virtual void draw( M3dView&, const MDagPath&, M3dView::DisplayStyle, M3dView::DisplayStatus );
		virtual MBoundingBox boundingBox() const;
		virtual bool isBounded() const;

	public:

		void draw( int drawingMode=0 );

    private:

        void autoConnect();

};

class BoraTestDrawOverride : public Bora_DrawOverride<BoraTest>
{
    public:

        BoraTestDrawOverride( const MObject& dgNodeObj )
        : Bora_DrawOverride<BoraTest>( dgNodeObj )
        {}
};

#endif

