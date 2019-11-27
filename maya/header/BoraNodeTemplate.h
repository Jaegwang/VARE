//--------------------//
// BoraNodeTemplate.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.01.24                               //
//-------------------------------------------------------//

#ifndef _BoraNodeTemplate_h_
#define _BoraNodeTemplate_h_

#include <Bora.h>
#include <MayaCommon.h>
#include <MayaUtils.h>
#include <Bora_ViewportDraw.h>

class BoraNodeTemplate : public MPxLocatorNode
{
	private:

        bool isThe1stTimeDraw = true;
		bool toUpdateAE = false;

		MObject           nodeObj;
		MString           nodeName;
		MFnDagNode        dagNodeFn;
        MDataBlock*       blockPtr;
		MFnDependencyNode nodeFn;

	public:

		static MTypeId id;
		static MString name;
		static MString drawDbClassification;
		static MString drawRegistrantId;

    public:

		static MObject inTimeObj;
		static MObject inXFormObj;
		static MObject outputObj;

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

class BoraNodeTemplateDrawOverride : public Bora_DrawOverride<BoraNodeTemplate>
{
    public:

        BoraNodeTemplateDrawOverride( const MObject& dgNodeObj )
        : Bora_DrawOverride<BoraNodeTemplate>( dgNodeObj )
        {}
};

#endif

