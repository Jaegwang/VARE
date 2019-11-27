//---------------------//
// BoraVDBFromPoints.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2017.09.20                               //
//-------------------------------------------------------//

#ifndef _BoraVDBFromPoints_h_
#define _BoraVDBFromPoints_h_

#include <MayaCommon.h>

class BoraVDBFromPoints : MPxLocatorNode
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

        static MObject radiusFreqObj;
        static MObject radiusOffsetObj;
        static MObject radiusScaleObj;

        static MObject densityFreqObj;
        static MObject densityOffsetObj;
        static MObject densityScaleObj;

        static MObject radiusObj;
        static MObject voxelSizeObj;

        static MObject inDataObj;
        static MObject outputObj;

    public:

        static std::map<std::string, BoraVDBFromPoints*> instances;

        Vec3fArray* points=0;
        Vec3fArray* velocities=0;

        std::string uid;
        std::string file;
        float radius;
        float voxelSize;
        
    public:

        BoraVDBFromPoints();

		static void* creator();
		virtual void postConstructor();
        static MStatus initialize();
		static void destructor( MObject& node, void* data );        
		virtual MStatus connectionMade( const MPlug&, const MPlug&, bool );
		virtual MStatus compute( const MPlug&, MDataBlock& );

		void draw( int drawingMode=0, const MHWRender::MDrawContext* context=0 );
		virtual void draw( M3dView&, const MDagPath&, M3dView::DisplayStyle, M3dView::DisplayStatus );
		virtual bool isBounded() const;
        virtual MBoundingBox boundingBox() const;
};

#endif

