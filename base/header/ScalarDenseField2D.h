//----------------------//
// ScalarDenseField2D.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
//         Jaegwang Lim @ Dexter Studios                 //
//         Julie Jang @ Dexter Studios                   //
// last update: 2018.04.25                               //
//-------------------------------------------------------//

#ifndef _BoraScalarDenseField2D_h_
#define _BoraScalarDenseField2D_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

class ScalarDenseField2D : public DenseField2D<float>
{
    public:

        ScalarDenseField2D();

        float min() const;

        float max() const;

        void drawLevelset( float maxInsideDistance, float maxOutsideDistance ) const;

        void drawSmoke() const;
};

BORA_NAMESPACE_END

#endif

