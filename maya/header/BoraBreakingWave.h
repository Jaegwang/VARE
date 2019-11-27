//--------------------//
// BoraBreakingWave.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2017.12.19                               //
//-------------------------------------------------------//

#pragma once

#include <Bora.h>
#include <MayaCommon.h>
#include <MayaUtils.h>

class BoraBreakingWave : public MPxNode
{
	private:

		MObject           nodeObj;
		MString           nodeName;
		MFnDependencyNode nodeFn;
		MFnDagNode        dagNodeFn;
		MDataBlock*       dataBlockPtr;

	private:

		BreakingWaveDeformer breakingWave;

		PointArray inputPoints;
		PointArray deformedPoints;

    public:

        static MTypeId id;
        static MString name;

		static MObject inMeshObj;
		static MObject inProfileLinesObj;
		static MObject outMeshObj;
		static MObject widthObj;

    public:

        BoraBreakingWave();
        ~BoraBreakingWave();

		static  void*   creator();
		virtual void    postConstructor();
		static  MStatus initialize();
		virtual MStatus compute( const MPlug& plug, MDataBlock& data );
		virtual MStatus connectionMade( const MPlug& plug, const MPlug& otherPlug, bool asSrc );
};

