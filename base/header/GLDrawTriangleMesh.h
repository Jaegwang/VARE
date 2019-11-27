//----------------------//
// GLDrawTriangleMesh.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2018.01.15                               //
//-------------------------------------------------------//

#ifndef _BoraGLDrawTriangleMesh_h_
#define _BoraGLDrawTriangleMesh_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

class GLDrawTriangleMesh
{
    private:

        static GLProgram _constColorShader;
        static GLProgram _phongShader;
        /*.......*/

    public:

        void drawTriangleArray( const TriangleMesh& mesh );
        void drawQuadArray( const TriangleMesh& mesh );

        void initialize();
};

BORA_NAMESPACE_END

#endif

