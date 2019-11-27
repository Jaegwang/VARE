//---------------//
// GLDrawOcean.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.04.02                               //
//-------------------------------------------------------//

#ifndef _BoraGLDrawOcean_h_
#define _BoraGLDrawOcean_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

class GLDrawOcean
{
    private:

        enum VertexBuffer
        {
            POS_BUFFER, // position
            NRM_BUFFER, // normal
            WAV_BUFFER, // wave
            CRS_BUFFER, // crest
            IDX_BUFFER, // triangle indices

            NUM_VERTEX_BUFFERS // # of vertex buffers
        };

    private:

        size_t _numPoints   = 0;
        size_t _numVertices = 0;

        GLProgram glProgram; // OpenGL shader program

        GLuint vertexArrayId = 0;                  // VAO
        GLuint vertexBufferId[NUM_VERTEX_BUFFERS]; // VBOs

        std::string skyTextureFile;
        int hasSkyTexture = 0;
        GLuint skyTextureId = 0;

    public:

        GLDrawOcean();

        virtual ~GLDrawOcean();

        void reset();

        void draw
        (
            const OceanTileVertexData& oceanTileVertexData,
            const Vec3f&               cameraPosition,
            const Vec3f&               deepWaterColor,
            const Vec3f&               shallowWaterColor,
            const char*                skyTextureFile,
            const float&               glossiness,
            const float&               exposure,
            const int&                 showTangles,
            const Vec3f&               tangleColor
        );

    private:

        bool loadSkyTexture( const char* filePathName );
};

BORA_NAMESPACE_END

#endif

