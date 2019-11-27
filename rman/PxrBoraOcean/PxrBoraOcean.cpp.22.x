//----------//
// main.cpp //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.04.24                               //
//-------------------------------------------------------//

// Refer:
// /netapp/backstage/pub/apps/renderman2/applications/linux/PixarRenderMan-Examples-22.0_1873521-linuxRHEL7_gcc48icc170.x86_64/plugins/pattern/thirdparty/aaOcean/aaOceanPrmanShader.args

#include <Bora.h>

#include <RixPredefinedStrings.hpp>
#include <RixPattern.h>
#include <RixShadingUtils.h>

BORA_NAMESPACE_BEGIN

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Class Declaration
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

class PxrBoraOcean : public RixPattern
{
    public:

        PxrBoraOcean() {};

        virtual int Init( RixContext&, RtUString const );
        virtual RixSCParamInfo const* GetParamTable();
        virtual void Finalize( RixContext& );
        virtual int ComputeOutputParams( RixShadingContext const*, RtInt*, RixPattern::OutputSpec**, RtPointer, RixSCParamInfo const* );
        virtual int CreateInstanceData(RixContext&, RtUString const, RixParameterList const*, InstanceData* );

        // newly addef for RenderMan ver 22.x
        virtual void Synchronize( RixContext&, RixSCSyncMsg, RixParameterList const* ) {}
        virtual bool Bake2dOutput( RixBakeContext const*, Bake2dSpec&, RtPointer ) { return false; }
        virtual bool Bake3dOutput( RixBakeContext const*, Bake3dSpec&, RtPointer ) { return false; }

    private:

        bool failed = false;

        RixTexture* m_tex = NULL;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Creator & Destroyer
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

RIX_PATTERNCREATE
{
    return new PxrBoraOcean();
}

RIX_PATTERNDESTROY
{
    delete ( (PxrBoraOcean*)pattern );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Parameter Table: Inputs & Outputs
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum ParameterIDs
{
    outputRGB_ID,
    outputA_ID,
    inputFile_ID,
    frameOffset_ID,
    rotationAngle_ID,
    scaleInX_ID,
    scaleInZ_ID,
    verticalScale_ID,
    horizontalScale_ID,
    crestGain_ID,
    crestBias_ID,
    enableLoop_ID,
    k_numParams
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Utility Functions
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T>
inline T* Malloc( RixShadingContext::Allocator& pool, RtInt n )
{
   return pool.AllocForPattern<T>( n );
}

inline RtInt NumOutputs( RixSCParamInfo const* paramTable )
{
    RtInt count = 0;
    while( paramTable[count++].access == k_RixSCOutput ) {}
    return count;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// PxrBoraOcean::Init()
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

int PxrBoraOcean::Init( RixContext& ctx, RtUString const pluginpath )
{
    RixMessages* msg = (RixMessages*)ctx.GetRixInterface( k_RixMessages );
    if( !msg ) { return 1; }

    m_tex = (RixTexture*)ctx.GetRixInterface( k_RixTexture );
    if( !m_tex ) { return 1; }

    return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// PxrBoraOcean::CreateInstanceData()
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

static void CleanUpFunc( void* oceanTileVertexDataData )
{
    OceanTileVertexData* oceanTileVertexData = (OceanTileVertexData*)oceanTileVertexDataData;
    delete oceanTileVertexData;
    oceanTileVertexData = NULL;
}

int PxrBoraOcean::CreateInstanceData( RixContext& ctx, RtUString handle, RixParameterList const* params, InstanceData* instance )
{
    // .oceanParams file path & name
    RtUString inputFile;
    params->EvalParam( inputFile_ID, -1, &inputFile );

    if( DoesFileExist( inputFile.CStr() ) == false )
    {
        COUT << "Error@PxrBoraOcean: Failed to get the input file." << ENDL;
        failed = true;
        return 1;
    }

    JSONString js;

    if( js.load( inputFile.CStr() ) == false )
    {
        COUT << "Error@PxrBoraOcean: Failed to load json file." << ENDL;
        failed = true;
        return 1;
    }

    std::string filePath;
    {
        std::vector<std::string> tokens;
        Split( std::string(inputFile.CStr()), "/", tokens );

        for( size_t i=0; i<tokens.size()-1; ++i )
        {
            filePath += "/" + tokens[i];
        }
    }

    std::string fileName;
    if( js.get( "fileName", fileName ) == false )
    {
        COUT << "Error@PxrBoraOcean: Failed to get fileName." << ENDL;
        failed = true;
        return 1;
    }

    float physicalLength = 0.f;
    if( js.get( "physicalLength", physicalLength ) == false )
    {
        COUT << "Error@PxrBoraOcean: Failed to get physicalLength." << ENDL;
        failed = true;
        return 1;
    }

    float sceneConvertingScale = 0.f;
    if( js.get( "sceneConvertingScale", sceneConvertingScale ) == false )
    {
        COUT << "Error@PxrBoraOcean: Failed to get sceneConvertingScale." << ENDL;
        failed = true;
        return 1;
    }

    // geometric length
    const float L = physicalLength * sceneConvertingScale;

    // the current frame number (currentFrame)
    RixRenderState* rs = (RixRenderState*)ctx.GetRixInterface( k_RixRenderState );
    RixRenderState::FrameInfo finfo;
    rs->GetFrameInfo( &finfo );
    const int currentFrame = (int)finfo.frame;

    int frameOffset = 0;
    params->EvalParam( frameOffset_ID, -1, &frameOffset );

    int enableLoop = 0;
    params->EvalParam( enableLoop_ID, -1, &enableLoop );

    int frameNo = currentFrame + frameOffset;
    {
        if( enableLoop )
        {
            std::vector<std::string> exr_files;
            GetFileList( filePath.c_str(), "exr", exr_files, false );
        
            std::sort( exr_files.begin(), exr_files.end() );

            const int num_exr_files = (int)exr_files.size();

            int the_1st_frameNo = 0;
            {
                std::vector<std::string> tokens;
                Split( exr_files[0], ".exr", tokens );

                the_1st_frameNo = atoi( tokens[tokens.size()-1].c_str() );
            }

            frameNo = ( frameNo - the_1st_frameNo ) % num_exr_files + the_1st_frameNo;
        }
    }

    const std::string paddedFrame = MakePadding( frameNo, 4 );

    std::stringstream ss;
    ss << filePath << "/" << fileName << "." << paddedFrame << ".exr";
    const std::string exrFile = ss.str();

    OceanTileVertexData* oceanTileVertexData = new OceanTileVertexData;

    if( oceanTileVertexData->importFromEXR( exrFile.c_str(), L ) )
    {
        failed = false;
    }
    else
    {
        std::cout << "[PxrBoraOcean] Failed to open file: " << exrFile << std::endl;
        failed = true;
    }

    instance->datalen  = sizeof(OceanTileVertexData*);
    instance->data     = (void*)oceanTileVertexData;
    instance->freefunc = CleanUpFunc;

    return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// PxrBoraOcean::GetParamTable()
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

RixSCParamInfo const* PxrBoraOcean::GetParamTable()
{
    static RixSCParamInfo s_ptable[] = 
    {
        // outputs
        RixSCParamInfo( RtUString("outputRGB"), k_RixSCColor, k_RixSCOutput ),
        RixSCParamInfo( RtUString("outputA"),   k_RixSCFloat, k_RixSCOutput ),

        // inputs
        RixSCParamInfo( RtUString("inputFile"),       k_RixSCString  ),
        RixSCParamInfo( RtUString("frameOffset"),     k_RixSCInteger ),
        RixSCParamInfo( RtUString("rotationAngle"),   k_RixSCFloat   ),
        RixSCParamInfo( RtUString("scaleInX"),        k_RixSCFloat   ),
        RixSCParamInfo( RtUString("scaleInZ"),        k_RixSCFloat   ),
        RixSCParamInfo( RtUString("verticalScale"),   k_RixSCFloat   ),
        RixSCParamInfo( RtUString("horizontalScale"), k_RixSCFloat   ),
        RixSCParamInfo( RtUString("crestGain"),       k_RixSCFloat   ),
        RixSCParamInfo( RtUString("crestBias"),       k_RixSCFloat   ),
        RixSCParamInfo( RtUString("enableLoop"),      k_RixSCInteger ),

        RixSCParamInfo() // end of table
    };

    return &s_ptable[0];
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// PxrBoraOcean::Finalize()
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

void PxrBoraOcean::Finalize( RixContext& ctx )
{
    // nothing to do
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// PxrBoraOcean::ComputeOutputParams(): The Main Process
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

int PxrBoraOcean::ComputeOutputParams
(
    RixShadingContext const* sctx,
    RtInt*                   howManyOutputs,
    OutputSpec**             outputSpec,
    RtPointer                instanceData,
    RixSCParamInfo const*    instanceTable
)
{
    if( failed ) { return 1; }

    const OceanTileVertexData* oceanTileVertexData = (OceanTileVertexData*)instanceData;

    const RtInt numOutputs = NumOutputs( GetParamTable() );

    const RtInt numShadedPoints = sctx->numPts;
    if( numShadedPoints <= 0 ) { return 1; }

    RtFloat const* rotationAngle;
    const RtFloat rotationAngleDefault = 0.f;
    sctx->EvalParam( rotationAngle_ID, -1, &rotationAngle, &rotationAngleDefault, false );

    RtFloat const* verticalScale;
    const RtFloat verticalScaleDefault = 1.f;
    sctx->EvalParam( verticalScale_ID, -1, &verticalScale, &verticalScaleDefault, false );

    RtFloat const* scaleInX;
    const RtFloat scaleInXDefault = 1.f;
    sctx->EvalParam( scaleInX_ID, -1, &scaleInX, &scaleInXDefault, false );

    RtFloat const* scaleInZ;
    const RtFloat scaleInZDefault = 1.f;
    sctx->EvalParam( scaleInZ_ID, -1, &scaleInZ, &scaleInZDefault, false );

    RtFloat const* horizontalScale;
    const RtFloat horizontalScaleDefault = 1.f;
    sctx->EvalParam( horizontalScale_ID, -1, &horizontalScale, &horizontalScaleDefault, false );

    RtFloat const* crestGain;
    const RtFloat crestGainDefault = 1.f;
    sctx->EvalParam( crestGain_ID, -1, &crestGain, &crestGainDefault, false );

    RtFloat const* crestBias;
    const RtFloat crestBiasDefault = 0.f;
    sctx->EvalParam( crestBias_ID, -1, &crestBias, &crestBiasDefault, false );

    RixShadingContext::Allocator pool( sctx );

    OutputSpec* outputData = Malloc<OutputSpec>( pool, numOutputs );
    *outputSpec = outputData;
    *howManyOutputs = numOutputs;

    for( RtInt i=0; i<numOutputs; ++i )
    {
        outputData[i].paramId = i;
        outputData[i].detail  = k_RixSCInvalidDetail;
        outputData[i].value   = NULL;

        RixSCType type;
        RixSCConnectionInfo cinfo;

        sctx->GetParamInfo( i, &type, &cinfo );

        if( cinfo == k_RixSCNetworkValue )
        {
            if( type == k_RixSCColor )
            {
                outputData[i].detail = k_RixSCVarying;
                outputData[i].value = Malloc<RtColorRGB>( pool, numShadedPoints );
            }
            else if( type == k_RixSCFloat )
            {
                outputData[i].detail = k_RixSCVarying;
                outputData[i].value = Malloc<RtFloat>( pool, numShadedPoints );
            }
        }
    }

    RtColorRGB* outputRGB = (RtColorRGB*)outputData[outputRGB_ID].value;
    if( !outputRGB ) { outputRGB = Malloc<RtColorRGB>( pool, numShadedPoints ); }

    RtFloat* outputA = (RtFloat*)outputData[outputA_ID].value;
    if( !outputA ) { outputA = Malloc<RtFloat>( pool, numShadedPoints ); }

    // world positions
    Vec3f* wP = NULL;
    {
        RtPoint3 const* P;
        sctx->GetBuiltinVar( RixShadingContext::k_Po, &P );

        wP = Malloc<Vec3f>( pool, numShadedPoints );
        memcpy( wP, P, numShadedPoints*sizeof(RtPoint3) );

        sctx->Transform( RixShadingContext::k_AsPoints, RtUString("current"), RtUString("world"), (RtPoint3*)wP );
    }

    // map scale
    for( RtInt i=0; i<numShadedPoints; ++i )
    {
        wP[i].x /= ( *scaleInX + 1e-6f );
        wP[i].z /= ( *scaleInZ + 1e-6f );
    }

    // map rotation
    float sn=0.f, cs=0.f;
    RixSinCos( -RixDegreesToRadians(*rotationAngle), &sn, &cs );

    if( *rotationAngle > 1e-6f )
    {
        for( RtInt i=0; i<numShadedPoints; ++i )
        {
            float& x = wP[i].x;
            float& z = wP[i].z;

            const float rx = ( x * cs ) - ( z * sn );
            const float rz = ( x * sn ) + ( z * cs );

            x = rx;
            z = rz;
        }
    }

    sn = -sn;

    // interpolation
    Vec3f G; // OceanTileVertexData::GRD
    Vec3f P; // OceanTileVertexData::POS
    float C; // OceanTileVertexData::CRS

    for( RtInt i=0; i<numShadedPoints; ++i ) 
    {
        oceanTileVertexData->lerp( wP[i], &G, &P, NULL, NULL, &C );

        Vec3f displacement = P - G;

        // displacement rotation
        if( *rotationAngle > 1e-6f )
        {
            float& x = displacement.x;
            float& z = displacement.z;

            const float rx = ( x * cs ) - ( z * sn );
            const float rz = ( x * sn ) + ( z * cs );

            x = rx;
            z = rz;
        }

        // displacement vector
        if( outputRGB )
        {
            outputRGB[i].r = displacement.x * (*horizontalScale);
            outputRGB[i].g = displacement.y * (*verticalScale);
            outputRGB[i].b = displacement.z * (*horizontalScale);
        }

        // crest value
        if( outputA )
        {
            outputA[i] = ( C * (*crestGain) ) + (*crestBias);
        }
    }

    return 0;
}

BORA_NAMESPACE_END

