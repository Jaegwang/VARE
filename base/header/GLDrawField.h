//---------------//
// FieldUtils.cu //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2017.11.20                               //
//-------------------------------------------------------//

#ifndef _BoraGLDrawField_h_
#define _BoraGLDrawField_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

class GLDrawField
{
    private:

        /* Shaders */
        static bool _initial;

    public:

        GLDrawField();

        void initialize();

        void drawGridBox( const Grid& grid );
        
        void drawVelocityField( const VectorDenseField& field, const float dt=0.01f );
        void drawDensityField( const ScalarDenseField& field );        
};

BORA_NAMESPACE_END

#endif

