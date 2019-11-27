//-------------//
// GLProgram.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2018.04.20                               //
//-------------------------------------------------------//

#ifndef _BoraGLProgram_h_
#define _BoraGLProgram_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

class GLProgram
{
    private:

        GLuint _id = 0; // program ID
        std::vector<GLuint> _shaderId; // shader IDs

    public:

        GLProgram();

        virtual ~GLProgram();

        void reset();

        bool addShader( const GLenum type, const char* source );

        bool link();

        GLuint id() const;

        void enable() const;
        void disable() const;

        GLuint attributeLocation( const char* name );
        GLuint uniformLocation( const char* name );

        bool bindModelViewMatrix( const char* name, bool transpose=false ) const;
        bool bindProjectionMatrix( const char* name, bool transpose=false ) const;

        bool bind( const char* name, const int&    v ) const;
        bool bind( const char* name, const float&  v ) const;
        bool bind( const char* name, const double& v ) const;
        bool bind( const char* name, const Vec2f&  v ) const;
        bool bind( const char* name, const Vec3f&  v ) const;
        bool bind( const char* name, const Vec4f&  v ) const;

		bool bindTexture( const char* name, GLuint textureId, GLenum type ) const;
};

BORA_NAMESPACE_END

#endif

