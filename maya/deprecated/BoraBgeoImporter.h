//--------------------//
// BoraBgeoImporter.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2017.09.15                               //
//-------------------------------------------------------//

#ifndef _BoraBgeoImpoter_h_
#define _BoraBgeoImpoter_h_

#include <MayaCommon.h>

class BoraBgeoImporter : MPxLocatorNode
{
    private:

        MObject           nodeObj;
        MString           nodeName;
        MFnDependencyNode nodeFn;
        MFnDagNode        dagNodeFn;
        MDataBlock*       dataBlockPtr=0;
        bool              is1stTime=true;

    public:

		static MTypeId id;
        static MString name;
		static MString drawDbClassification;
        static MString drawRegistrantId;

        static MObject cachePathObj;
        static MObject inTimeObj;
        static MObject outDataObj;

    public:
        // houdini bgeo attributes
        Vec3fArray point_p;
        Vec3fArray point_v;
        
        std::string bgeoFile;
        
    public:

        BoraBgeoImporter();

		static void* creator();
		virtual void postConstructor();
        static MStatus initialize();
		virtual MStatus connectionMade( const MPlug&, const MPlug&, bool );        
		virtual MStatus compute( const MPlug&, MDataBlock& );

		void draw( int drawingMode=0, const MHWRender::MDrawContext* context=0 );
		virtual void draw( M3dView&, const MDagPath&, M3dView::DisplayStyle, M3dView::DisplayStatus );
		virtual bool isBounded() const;
        virtual MBoundingBox boundingBox() const;
};

#endif

