//--------------------//
// VEX_OceanSampler.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.11.23                               //
//-------------------------------------------------------//

#pragma once
#include <HouCommon.h>

class VEX_OceanSampler
{
    public:

        static std::atomic_flag _boraOceanLock;

        static const char* signature;

        struct BoraOceanTrunk
        {
            OceanTileVertexData boraOceanTileVertexData;
            int boraOceanCurrentFrame = -1;
            bool isGood = true;
        };

    public:

        static void* init()
        {
            return ( void* )new BoraOceanTrunk();
        }

        static void clear( void* data )
        {
            if( data ) delete (BoraOceanTrunk*)data;
        }

        static void compute( int argc, void* argv[], void* data )
        {
            BoraOceanTrunk* trunk = (BoraOceanTrunk*)data;

            UT_Vector3* out = (UT_Vector3*)argv[0];

            const UT_Vector3 pos = *((UT_Vector3*)argv[1]);
            const int frameNo = *((float*)argv[2]);
            const char* json_path = (const char*)argv[3];

            std::string ss(json_path);
            if( ss.length() == 0 )
            {
                (*out) = UT_Vector3( 0.f, 0.f, 0.f );
                return;
            }

            while( _boraOceanLock.test_and_set() );

            if( trunk->boraOceanCurrentFrame != frameNo )
            {
                trunk->boraOceanCurrentFrame = frameNo;

                JSONString js;
                if( js.load( json_path ) == false ) trunk->isGood = false;

                std::string filePath;
                {
                    std::vector<std::string> tokens;
                    Split( std::string(json_path), "/", tokens );

                    for( size_t i=0; i<tokens.size()-1; ++i )
                    {
                        filePath += "/" + tokens[i];
                    }
                }

                float physicalLength = 0.f;
                if( js.get( "physicalLength", physicalLength ) == false ) trunk->isGood = false;

                float sceneConvertingScale = 0.f;
                if( js.get( "sceneConvertingScale", sceneConvertingScale ) == false ) trunk->isGood = false;

                const float L = physicalLength*sceneConvertingScale;        

                std::string fileName;
                if( js.get( "fileName", fileName ) == false ) trunk->isGood = false;


                if( trunk->isGood == true )
                {            
                    const std::string paddedFrame = MakePadding( frameNo, 4 );

                    std::stringstream ss;
                    ss << filePath << "/" << fileName << "." << paddedFrame << ".exr";
                    const std::string exrFile = ss.str();

                    std::ifstream f( exrFile.c_str() );
                    if( f.good() )
                        trunk->boraOceanTileVertexData.importFromEXR( exrFile.c_str(), L );
                    else
                        trunk->isGood = false;
                }
            }

            _boraOceanLock.clear();

            // OUT Pos
            if( trunk->isGood == true )
            {
                Vec3f wP( pos.x(), pos.y(), pos.z() );
                Vec3f G, P;
                float C;

                trunk->boraOceanTileVertexData.lerp( wP, &G, &P, NULL, NULL, &C );
                Vec3f disp = P - G;

                *out = UT_Vector3( disp.x, disp.y, disp.z );
            }
            else
            {
                *out = UT_Vector3( 0.f, 0.f, 0.f );
            }
        }

};

const char* VEX_OceanSampler::signature = "bora_ocean_sampler@&VVFS";
std::atomic_flag VEX_OceanSampler::_boraOceanLock = ATOMIC_FLAG_INIT;
