//------------------------//
// GLDrawTriangleMesh.cpp //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2017.11.06                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

GLProgram GLDrawTriangleMesh::_constColorShader;
GLProgram GLDrawTriangleMesh::_phongShader;

void GLDrawTriangleMesh::drawTriangleArray( const TriangleMesh& mesh )
{
    const Vec3fArray& pos = mesh.position;
    
    _constColorShader.enable();
    
    _constColorShader.bindModelViewMatrix( "uModelViewMatrix" );
    _constColorShader.bindProjectionMatrix( "uProjectionMatrix" );

    GLuint aPos = _constColorShader.attributeLocation( "aPos" );

    glVertexAttribPointer( aPos, 3, GL_FLOAT, false, 0, &(pos.at(0)) );
    
    glEnableVertexAttribArray( aPos );

    glDrawArrays( GL_TRIANGLES, 0, pos.num() );

    glDisableVertexAttribArray( aPos );        
    
    _constColorShader.disable();
}

void GLDrawTriangleMesh::drawQuadArray( const TriangleMesh& mesh )
{

}

void GLDrawTriangleMesh::initialize()
{
    _constColorShader.addShader( GL_VERTEX_SHADER, (const char*)STRINGER(
        
        \n#version 450\n
        
        layout(location=0) in vec3 aPos;
    
        uniform mat4 uProjectionMatrix;
        uniform mat4 uModelViewMatrix;

        void main()
        {
            gl_Position = uProjectionMatrix * uModelViewMatrix * vec4(aPos, 1.0);
        }

    ));
        
    _constColorShader.addShader( GL_FRAGMENT_SHADER, (const char*)STRINGER(

        \n#version 450\n
        
        out vec4 outColor;

        void main()
        {
            outColor = vec4( 1.0, 1.0, 0.0, 1.0 );
        }

    ));

    _constColorShader.link();
}

BORA_NAMESPACE_END

