//-------------//
// BoraOcean.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.12.20                               //
//-------------------------------------------------------//

#ifndef _BoraOcean_h_
#define _BoraOcean_h_

#include <Bora.h>
#include <MayaCommon.h>
#include <MayaUtils.h>
#include <Bora_ViewportDraw.h>

class BoraOcean : public MPxLocatorNode
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

        int _gridLevel = -1;

        OceanTile           oceanTile;
        OceanTileVertexData oceanTileVertexData;
        GLDrawOcean         glDrawOcean;
        TriangleMesh        inputMesh;
        AABB3f              aabb;
        int                 i0 = 0, i1 = 1;
        int                 j0 = 0, j1 = 1;

	public:

		static MTypeId id;
		static MString name;
		static MString drawDbClassification;
		static MString drawRegistrantId;

    public:

		static MObject inTimeObj;
		static MObject inMeshObj;
		static MObject outputObj;
        static MObject dispersionObj;
        static MObject spectrumObj;
        static MObject spreadingObj;
        static MObject filterObj;
        static MObject randomObj;
        static MObject numThreadsObj;
        static MObject grainSizeObj;
        static MObject gridLevelObj;
        static MObject physicalLengthObj;
        static MObject sceneConvertingScaleObj;
        static MObject gravityObj;
        static MObject surfaceTensionObj;
        static MObject densityObj;
        static MObject depthObj;
        static MObject windDirectionObj;
        static MObject windSpeedObj;
        static MObject flowSpeedObj;
        static MObject fetchObj;
        static MObject swellObj;
        static MObject filterSoftWidthObj;
        static MObject filterSmallWavelengthObj;
        static MObject filterBigWavelengthObj;
        static MObject randomSeedObj;
        static MObject crestGainObj;
        static MObject crestBiasObj;
        static MObject crestAccumulationObj;
        static MObject crestDecayObj;
        static MObject timeOffsetObj;
        static MObject timeScaleObj;
        static MObject loopingDurationObj;
        static MObject amplitudeGainObj;
        static MObject pinchObj;
        static MObject mapResolutionObj;
        static MObject displayModeObj;
        static MObject deepWaterColorObj;
        static MObject shallowWaterColorObj;
        static MObject skyTextureFileObj;
        static MObject glossinessObj;
        static MObject exposureObj;
        static MObject drawOutlineObj;
        static MObject outlineColorObj;
        static MObject outlineWidthObj;
        static MObject showTanglesObj;
        static MObject tangleColorObj;
        static MObject onOffObj;

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
        void calcTileRange();

    public:

        const OceanParams& getOceanParams() const;
        const OceanTile& getOceanTile() const;
        const OceanTileVertexData& getOceanTileVertexData() const;
};

class BoraOceanDrawOverride : public Bora_DrawOverride<BoraOcean>
{
    public:

        BoraOceanDrawOverride( const MObject& dgNodeObj )
        : Bora_DrawOverride<BoraOcean>( dgNodeObj )
        {}
};

#endif

