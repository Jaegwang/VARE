//------------//
// BoraGrid.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.01.24                               //
//-------------------------------------------------------//

#ifndef _BoraGrid_h_
#define _BoraGrid_h_

#include <Bora.h>
#include <MayaCommon.h>
#include <MayaUtils.h>
#include <Bora_ViewportDraw.h>

class BoraGrid : public MPxLocatorNode
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

        Grid grid;

        MString scalarFieldMemorySize;
        MString vectorFieldMemorySize;

	public:

		static MTypeId id;
		static MString name;
		static MString drawDbClassification;
		static MString drawRegistrantId;

    public:

        static MObject inputObj;
        static MObject inXFormObj;
        static MObject outputObj;

        static MObject subdivisionObj;

        static MObject displayGridObj;
        static MObject dispGridX0Obj;
        static MObject dispGridX1Obj;
        static MObject dispGridY0Obj;
        static MObject dispGridY1Obj;
        static MObject dispGridZ0Obj;
        static MObject dispGridZ1Obj;

        static MObject resolutionObj;
        static MObject scalarFieldMemorySizeObj;
        static MObject vectorFieldMemorySizeObj;

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

class BoraGridDrawOverride : public Bora_DrawOverride<BoraGrid>
{
    public:

        BoraGridDrawOverride( const MObject& dgNodeObj )
        : Bora_DrawOverride<BoraGrid>( dgNodeObj )
        {}
};

#endif

