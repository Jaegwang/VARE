//-------------------//
// GLDrawParticles.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2017.11.02                               //
//-------------------------------------------------------//

#ifndef _BoraGLDrawParticles_h_
#define _BoraGLDrawParticles_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

class GLDrawParticles
{
    private:

        static GLProgram _constColorShader;        
        static GLProgram _velocityBlendedShader;
        static bool _initial;

    public:

        GLDrawParticles();

        void initialize();        

        void draw( const Particles& pts );
        void drawVelocities( const Particles& pts, Vec3f minColor, Vec3f maxColor, float minV, float maxV );
};

BORA_NAMESPACE_END

#endif

