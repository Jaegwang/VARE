//--------------------//
// VectorDenseField.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
//         Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.01.15                               //
//-------------------------------------------------------//

#ifndef _BoraVectorDenseField_h_
#define _BoraVectorDenseField_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

class VectorDenseField : public DenseField<Vec3f>
{
    public:

        VectorDenseField();

		float maxMagnitude( int& idx ) const;
};

BORA_NAMESPACE_END

#endif

