//-------------------//
// IndexDenseField.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
//         Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.01.15                               //
//-------------------------------------------------------//

#ifndef _BoraIndexDenseField_h_
#define _BoraIndexDenseField_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

class IndexDenseField : public DenseField<size_t>
{
    public:

        IndexDenseField();
};

BORA_NAMESPACE_END

#endif

