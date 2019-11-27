//-------------------//
// BoraHeightMerge.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2019.03.19                               //
//-------------------------------------------------------//

#pragma once

#include <Bora.h>
#include <MayaCommon.h>
#include <MayaUtils.h>
#include <Bora_ViewportDraw.h>

class BoraHeightMerge : public MPxLocatorNode
{
	private:

		MObject           nodeObj;
		MString           nodeName;
		MFnDependencyNode nodeFn;
		MFnDagNode        dagNodeFn;
		MDataBlock*       dataBlockPtr;

	private:

		PointArray vertices;
		IntArray   triangles;

        Vec2f min = Vec2f(  1e+10f );
        Vec2f max = Vec2f( -1e+10f );
		float voxelSize;

		HashGrid2D hashGrid;

		bool rayTriangleIntersection( Vec3f& P, const Vec3f& P0, const Vec3f& V, const Vec3f& T0, const Vec3f& T1, const Vec3f& T2 );

    public:

        static MTypeId id;
        static MString name;
		static MString drawDbClassification;
		static MString drawRegistrantId;		

		static MObject inHeightMeshsObj;
		static MObject inDomainMeshObj;
		static MObject outMeshObj;

    public:

        BoraHeightMerge();
        ~BoraHeightMerge();

		static  void*   creator();
		virtual void    postConstructor();
		static  MStatus initialize();
		virtual MStatus compute( const MPlug& plug, MDataBlock& data );
		virtual MStatus connectionMade( const MPlug& plug, const MPlug& otherPlug, bool asSrc );

		virtual void draw( M3dView&, const MDagPath&, M3dView::DisplayStyle, M3dView::DisplayStatus );
		void draw( int drawingMode=0 );

		virtual MBoundingBox boundingBox() const;
};

class BoraHeightMergeDrawOverride : public Bora_DrawOverride<BoraHeightMerge>
{
    public:

        BoraHeightMergeDrawOverride( const MObject& dgNodeObj )
        : Bora_DrawOverride<BoraHeightMerge>( dgNodeObj )
        {}
};
