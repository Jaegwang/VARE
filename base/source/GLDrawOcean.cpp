//-----------------//
// GLDrawOcean.cpp //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.04.02                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

GLDrawOcean::GLDrawOcean()
{
    // nothing to do
}

GLDrawOcean::~GLDrawOcean()
{
    GLDrawOcean::reset();
}

void GLDrawOcean::reset()
{
    if( vertexArrayId > 0 )
    {
        glDeleteVertexArrays( 1, &vertexArrayId );
        vertexArrayId = 0;

        glDeleteBuffers( NUM_VERTEX_BUFFERS, vertexBufferId );
    }

    if( skyTextureId > 0 )
    {
        glDeleteTextures( 1, &skyTextureId );
    }
}

void GLDrawOcean::draw
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
)
{
    if( glProgram.id() == 0 )
    {
        glProgram.addShader( GL_VERTEX_SHADER,   OceanVS );
        glProgram.addShader( GL_FRAGMENT_SHADER, OceanFS );
        glProgram.link();
    }

    const Vec3fArray& POS = oceanTileVertexData.POS;
    const Vec3fArray& NRM = oceanTileVertexData.NRM;
    const FloatArray& WAV = oceanTileVertexData.WAV;
    const FloatArray& CRS = oceanTileVertexData.CRS;
    const UIntArray&  TRI = oceanTileVertexData.TRI;

    const int numPoints   = (int)POS.size();
    const int numVertices = (int)TRI.size();

    if( ( vertexArrayId == 0 ) || ( _numPoints != numPoints ) || ( _numVertices != numVertices ) )
    {
        GLDrawOcean::reset();

        glGenVertexArrays( 1, &vertexArrayId );

        glBindVertexArray( vertexArrayId );
        {
            glGenBuffers( NUM_VERTEX_BUFFERS, vertexBufferId );

            glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, vertexBufferId[IDX_BUFFER] );
            glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int)*numVertices, &(TRI[0]), GL_STATIC_DRAW );
        }
        glBindVertexArray( 0 );

        _numPoints   = numPoints;
        _numVertices = numVertices;
    }

    if( vertexArrayId > 0 )
    {
        const OceanTile& oceanTile = oceanTileVertexData.oceanTile();
        const float& minHeight = oceanTileVertexData.minHeight;
        const float& maxHeight = oceanTileVertexData.maxHeight;

        glProgram.enable();
        {
            glProgram.bindModelViewMatrix( "modelViewMatrix" );
            glProgram.bindProjectionMatrix( "projectionMatrix" );

            glProgram.bind( "cameraPosition",    cameraPosition    );
            glProgram.bind( "deepWaterColor",    deepWaterColor    );
            glProgram.bind( "shallowWaterColor", shallowWaterColor );
            glProgram.bind( "minHeight",         minHeight         );
            glProgram.bind( "maxHeight",         maxHeight         );
            glProgram.bind( "glossiness",        glossiness        );
            glProgram.bind( "exposure",          exposure          );
            glProgram.bind( "showTangles",       showTangles       );
            glProgram.bind( "tangleColor",       tangleColor       );

            hasSkyTexture = loadSkyTexture( skyTextureFile );
            glProgram.bind( "hasSkyTexture", hasSkyTexture );
            glProgram.bindTexture( "skyTexture", skyTextureId, GL_TEXTURE_2D );

            glBindVertexArray( vertexArrayId );
            {
                glBindBuffer( GL_ARRAY_BUFFER, vertexBufferId[POS_BUFFER] );
                glBufferData( GL_ARRAY_BUFFER, sizeof(Vec3f)*numPoints, &(POS[0]), GL_DYNAMIC_DRAW );
                glEnableVertexAttribArray( POS_BUFFER );
                glVertexAttribPointer( POS_BUFFER, 3, GL_FLOAT, GL_FALSE, 0, 0 );

                glBindBuffer( GL_ARRAY_BUFFER, vertexBufferId[NRM_BUFFER] );
                glBufferData( GL_ARRAY_BUFFER, sizeof(Vec3f)*numPoints, &(NRM[0]), GL_DYNAMIC_DRAW );
                glEnableVertexAttribArray( NRM_BUFFER );
                glVertexAttribPointer( NRM_BUFFER, 3, GL_FLOAT, GL_FALSE, 0, 0 );

                glBindBuffer( GL_ARRAY_BUFFER, vertexBufferId[WAV_BUFFER] );
                glBufferData( GL_ARRAY_BUFFER, sizeof(float)*numPoints, &(WAV[0]), GL_DYNAMIC_DRAW );
                glEnableVertexAttribArray( WAV_BUFFER );
                glVertexAttribPointer( WAV_BUFFER, 1, GL_FLOAT, GL_FALSE, 0, 0 );

                glBindBuffer( GL_ARRAY_BUFFER, vertexBufferId[CRS_BUFFER] );
                glBufferData( GL_ARRAY_BUFFER, sizeof(float)*numPoints, &(CRS[0]), GL_DYNAMIC_DRAW );
                glEnableVertexAttribArray( CRS_BUFFER );
                glVertexAttribPointer( CRS_BUFFER, 1, GL_FLOAT, GL_FALSE, 0, 0 );

                glDrawElements( GL_TRIANGLES, numVertices, GL_UNSIGNED_INT, 0 );
            }
            glBindVertexArray( 0 );

            glBindTexture( GL_TEXTURE_2D, 0 );
        }
        glProgram.disable();
    }
}

bool GLDrawOcean::loadSkyTexture( const char* filePathName )
{
    if( skyTextureFile == filePathName )
    {
        return true;
    }

    skyTextureFile = filePathName;

    if( skyTextureId > 0 )
    {
        glDeleteTextures( 1, &skyTextureId );
    }

    glGenTextures( 1, &skyTextureId );

    Image img;

    if( img.load( filePathName ) == false )
    {
        COUT << "Error@GLDrawOcean::loadSkyTexture(): Failed to load the image." << ENDL;
        return false;
    }

    const Pixel* imgData = img.pointer();

    glBindTexture( GL_TEXTURE_2D, skyTextureId );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, img.width(), img.height(), 0, GL_RGBA, GL_HALF_FLOAT, imgData );
    glBindTexture( GL_TEXTURE_2D, 0 );

    return true;
}

BORA_NAMESPACE_END

