//----------------------//
// VectorDenseField2D.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
//         Jaegwang Lim @ Dexter Studios                 //
//         Julie Jang @ Dexter Studios                   //
// last update: 2018.04.25                               //
//-------------------------------------------------------//

#ifndef _BoraVectorDenseField2D_h_
#define _BoraVectorDenseField2D_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

class VectorDenseField2D : public DenseField2D<Vec2f>
{
    public:

        VectorDenseField2D();

		float maxMagnitude( int& idx ) const;

};

BORA_NAMESPACE_END

#endif

