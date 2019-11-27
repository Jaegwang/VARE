//-------------//
// GLCapture.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.04.19                               //
//-------------------------------------------------------//

#ifndef _BoraGLCapture_h_
#define _BoraGLCapture_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

class GLCapture
{
    private:

        GLsizei _width      = 0; // image width
        GLsizei _height     = 0; // image height

        GLuint  _fboId      = 0; // framebuffer object id
        GLuint  _colorTexId = 0; // color texture buffer object id
        GLuint  _depthRboId = 0; // depth render buffer object id

        half*   _pixels = nullptr;
        Image   _img;

    public:

        GLCapture();

        virtual ~GLCapture();

        void reset();

        bool initialize( GLsizei width, GLsizei height );

        void begin();
        void end();

        GLsizei width()  const;
        GLsizei height() const;
        GLenum  format() const;
        GLenum  type()   const;

        GLuint fboId()      const;
        GLuint colorTexId() const;
        GLuint depthRboId() const;

        bool save( const char* filePathName ) const;
};

BORA_NAMESPACE_END

#endif

