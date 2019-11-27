//---------------//
// GLProgram.cpp //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2018.04.20                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

GLProgram::GLProgram()
{
    // nothing to do
}

GLProgram::~GLProgram()
{
    GLProgram::reset();
}

void GLProgram::reset()
{
    for( size_t i=0; i<_shaderId.size(); ++i )
    {
        glDetachShader( _id, _shaderId[i] );
        glDeleteShader( _shaderId[i] );
    }

    _shaderId.clear();

    if( _id > 0 )
    {
        glDeleteProgram( _id );
        glUseProgram( 0 );
    }

    _id = 0;
}

// glCreateShader() -> glShaderSource() -> glCompileShader()
bool GLProgram::addShader( GLenum type, const char* source )
{
    if( _id > 0 )
    {
        COUT << "Error@GLProgram::addShader(): Too late." << ENDL;
        return false;
    }

    bool isValidType = false;
    if( type == GL_VERTEX_SHADER          ) { isValidType = true; }
    if( type == GL_FRAGMENT_SHADER        ) { isValidType = true; }
    if( type == GL_GEOMETRY_SHADER        ) { isValidType = true; }
    if( type == GL_TESS_CONTROL_SHADER    ) { isValidType = true; }
    if( type == GL_TESS_EVALUATION_SHADER ) { isValidType = true; }

    if( isValidType == false )
    {
        COUT << "Error@GLProgram::addShader(): Invalid shader type." << ENDL;
        return false;
    }

    const GLuint shaderId = glCreateShader( type );

    glShaderSource( shaderId, 1, &source, 0 );

    glCompileShader( shaderId );
    {
        GLint status = GL_TRUE;
        glGetShaderiv( shaderId, GL_COMPILE_STATUS, &status );
        
        if( status != GL_TRUE )
        {
            GLint logLength = 0;
            glGetShaderiv( shaderId, GL_INFO_LOG_LENGTH, &logLength );

            std::vector<GLchar> log( logLength );
            glGetShaderInfoLog( shaderId, logLength, &logLength, &log[0] );

            COUT << "Error@GLProgram::addShader(): " << &log[0] << ENDL;

            glDetachShader( _id, shaderId );
            glDeleteShader( shaderId );

            return false;
        }
    }

    _shaderId.push_back( shaderId );

    return true;
}

// glCreateProgram() -> glAttachShader() -> glLinkProgram()
bool GLProgram::link()
{
    if( _id > 0 ) { return true; }

    _id = glCreateProgram();

    if( _id == 0 )
    {
        COUT << "Error@GLProgram::link(): Failed to create GLSL program." << ENDL;
        return false;
    }

    for( size_t i=0; i<_shaderId.size(); ++i )
    {
        glAttachShader( _id, _shaderId[i] );
    }

    glLinkProgram( _id );
    {
        GLint status = 0;
        glGetProgramiv( _id, GL_LINK_STATUS, &status );

        if( status != GL_TRUE )
        {
            GLint logLength = 0;
            glGetProgramiv( _id, GL_INFO_LOG_LENGTH, &logLength );

            std::vector<GLchar> log( logLength+1 );
            glGetProgramInfoLog( _id, logLength, &logLength, &(log[0]) );

            std::string logMsg = (const std::string&)( &(log[0]) );
            COUT << "Error@GLProgram::link(): " << logMsg << ENDL;

            GLProgram::reset();

            return false;
        }
    }

    // Disable the programmable processors so that the fixed functionality will be used.
    glUseProgram( 0 );

    return true;
}

GLuint GLProgram::id() const
{
    return _id;
}

void GLProgram::enable() const
{
    glUseProgram( _id );
}

void GLProgram::disable() const
{
    glUseProgram( 0 );
}

GLuint GLProgram::attributeLocation( const char* name )
{
    return glGetAttribLocation( _id, name );
}

GLuint GLProgram::uniformLocation( const char* name )
{
    return glGetUniformLocation( _id, name );
}

bool GLProgram::bindModelViewMatrix( const char* name, bool transpose ) const
{
    const GLint location = glGetUniformLocation( _id, name );
	if( location < 0 ) { return false; }

    float matrix[16];
    glGetFloatv( GL_MODELVIEW_MATRIX, matrix );

   	glUniformMatrix4fv( location, 1, ( transpose ? GL_TRUE : GL_FALSE ), matrix );
    return true;
}

bool GLProgram::bindProjectionMatrix( const char* name, bool transpose ) const
{
    const GLint location = glGetUniformLocation( _id, name );
	if( location < 0 ) { return false; }

    float matrix[16];
    glGetFloatv( GL_PROJECTION_MATRIX, matrix );

   	glUniformMatrix4fv( location, 1, ( transpose ? GL_TRUE : GL_FALSE ), matrix );
    return true;
}

bool GLProgram::bind( const char* name, const int& v ) const
{
    const GLint location = glGetUniformLocation( _id, name );
	if( location < 0 ) { return false; }

    glUniform1i( location, v );
    return true;
}

bool GLProgram::bind( const char* name, const float& v ) const
{
    const GLint location = glGetUniformLocation( _id, name );
	if( location < 0 ) { return false; }

    glUniform1f( location, v );
    return true;
}

bool GLProgram::bind( const char* name, const double& v ) const
{
    const GLint location = glGetUniformLocation( _id, name );
	if( location < 0 ) { return false; }

    glUniform1f( location, (float)v );
    return true;
}

bool GLProgram::bind( const char* name, const Vec2f& v ) const
{
    const GLint location = glGetUniformLocation( _id, name );
	if( location < 0 ) { return false; }

    glUniform2f( location, v.x, v.y );
    return true;
}

bool GLProgram::bind( const char* name, const Vec3f& v ) const
{
    const GLint location = glGetUniformLocation( _id, name );
	if( location < 0 ) { return false; }

    glUniform3f( location, v.x, v.y, v.z );
    return true;
}

bool GLProgram::bind( const char* name, const Vec4f& v ) const
{
    const GLint location = glGetUniformLocation( _id, name );
	if( location < 0 ) { return false; }

    glUniform4f( location, v.x, v.y, v.z, v.w );
    return true;
}

bool GLProgram::bindTexture( const char* name, GLuint textureId, GLenum type ) const
{
	GLint location = glGetUniformLocation( _id, name );
	if( location < 0 ) { return false; }

	glActiveTexture( GL_TEXTURE0 + (textureId-1) );
	glBindTexture( type, textureId );
	glUniform1i( location, (textureId-1) );

    return true;
}

BORA_NAMESPACE_END

