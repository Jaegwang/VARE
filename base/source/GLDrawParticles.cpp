//---------------------//
// GLDrawParticles.cpp //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2017.11.01                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

GLProgram GLDrawParticles::_velocityBlendedShader;
GLProgram GLDrawParticles::_constColorShader;
bool GLDrawParticles::_initial = false;

GLDrawParticles::GLDrawParticles()
{
    if( _initial == true ) return;

    initialize();
    _initial = true;
}

void
GLDrawParticles::draw( const Particles& pts )
{
    const Vec3fArray& pos = pts.position;

    _constColorShader.enable();

    _constColorShader.bindModelViewMatrix( "uModelViewMatrix" );
    _constColorShader.bindProjectionMatrix( "uProjectionMatrix" );

    GLuint aPos = _constColorShader.attributeLocation( "aPos" );

    glVertexAttribPointer( aPos, 3, GL_FLOAT, false, 0, pos.pointer() );

    glEnableVertexAttribArray( aPos );

    glDrawArrays( GL_POINTS, 0, pos.num() );
    
    glDisableVertexAttribArray( aPos );

    _constColorShader.disable();
}

void
GLDrawParticles::drawVelocities( const Particles& pts, Vec3f minColor, Vec3f maxColor, float minV, float maxV )
{
    const Vec3fArray& pos = pts.position;
    const Vec3fArray& vel = pts.velocity;

    _velocityBlendedShader.enable();

    _velocityBlendedShader.bindModelViewMatrix( "uModelViewMatrix" );
    _velocityBlendedShader.bindProjectionMatrix( "uProjectionMatrix" );

    _velocityBlendedShader.bind( "minVel", minV );
    _velocityBlendedShader.bind( "maxVel", maxV );

    _velocityBlendedShader.bind( "minColor", minColor );
    _velocityBlendedShader.bind( "maxColor", maxColor );

    GLuint aPos = _velocityBlendedShader.attributeLocation( "aPos" );
    GLuint aVel = _velocityBlendedShader.attributeLocation( "aVel" );

    glVertexAttribPointer( aPos, 3, GL_FLOAT, false, 0, pos.pointer() );
    glVertexAttribPointer( aVel, 3, GL_FLOAT, false, 0, vel.pointer() );

    glEnableVertexAttribArray( aPos );
    glEnableVertexAttribArray( aVel );

    glDrawArrays( GL_POINTS, 0, pos.num() );
    
    glDisableVertexAttribArray( aVel );
    glDisableVertexAttribArray( aPos );

    _velocityBlendedShader.disable();
}

void
GLDrawParticles::initialize()
{
    _constColorShader.addShader( GL_VERTEX_SHADER, (const char*)STRINGER(

        \n#version 120\n
        
        attribute vec3 aPos;
    
        uniform mat4 uProjectionMatrix;
        uniform mat4 uModelViewMatrix;

        void main()
        {
            gl_Position = uProjectionMatrix * uModelViewMatrix * vec4(aPos, 1.0);
        }

    ));

    _constColorShader.addShader( GL_FRAGMENT_SHADER, (const char*)STRINGER(

        \n#version 120\n

        void main()
        {
            gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );
        }

    ));

    _constColorShader.link();

    _velocityBlendedShader.addShader( GL_VERTEX_SHADER, (const char*)STRINGER(
    
        \n#version 120\n
        
        attribute vec3 aPos;
        attribute vec3 aVel;
    
        uniform mat4 uProjectionMatrix;
        uniform mat4 uModelViewMatrix;

        varying vec3 vVel;

        void main()
        {
            vVel = aVel;
            gl_Position = uProjectionMatrix * uModelViewMatrix * vec4(aPos, 1.0);
        }
    ));

    _velocityBlendedShader.addShader( GL_FRAGMENT_SHADER, (const char*)STRINGER(

        \n#version 120\n

        uniform float minVel;
        uniform float maxVel;
        uniform vec3 minColor;
        uniform vec3 maxColor;

        varying vec3 vVel;

        void main()
        {
            float mag = length( vVel );
            
            float a = min( (mag-minVel)/(maxVel-minVel), 1.0 );

            vec3 color = maxColor*a + minColor*(1.0-a);

            gl_FragColor = vec4( color, 1.0 );
        }
    ));

    _velocityBlendedShader.link();
}

BORA_NAMESPACE_END

