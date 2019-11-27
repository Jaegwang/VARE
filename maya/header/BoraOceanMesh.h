//-----------------//
// BoraOceanMesh.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2019.01.23                               //
//-------------------------------------------------------//

#pragma once

#include <Bora.h>
#include <MayaCommon.h>
#include <MayaUtils.h>

class BoraOceanMesh : public MPxNode
{
	private:

		MObject           nodeObj;
		MString           nodeName;
		MFnDependencyNode nodeFn;
		MFnDagNode        dagNodeFn;
		MDataBlock*       dataBlockPtr;

	private:

		MPointArray meshPoints;
        MPointArray deformedPoints;

    public:

        static MTypeId id;
        static MString name;

		static MObject inOceanObj;//float
		static MObject inMeshObj;
		static MObject outMeshObj;


    public:

        BoraOceanMesh();
        ~BoraOceanMesh();

		static  void*   creator();
		virtual void    postConstructor();
		static  MStatus initialize();
		virtual MStatus compute( const MPlug& plug, MDataBlock& data );
		virtual MStatus connectionMade( const MPlug& plug, const MPlug& otherPlug, bool asSrc );
};

