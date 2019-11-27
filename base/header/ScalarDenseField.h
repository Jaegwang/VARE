//--------------------//
// ScalarDenseField.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
//         Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.01.15                               //
//-------------------------------------------------------//

#ifndef _BoraScalarDenseField_h_
#define _BoraScalarDenseField_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

class ScalarDenseField : public DenseField<float>
{
    public:

        ScalarDenseField();

        float min() const;
        float max() const;

        Vec3f gradient( const size_t& i, const size_t& j, const size_t& k );
        Vec3f gradient( const Idx3& idx );
};

BORA_NAMESPACE_END

#endif

