//----------------------//
// MarkerDenseField2D.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
//         Jaegwang Lim @ Dexter Studios                 //
//         Julie Jang @ Dexter Studios                   //
// last update: 2018.04.25                               //
//-------------------------------------------------------//

#ifndef _BoraMarkerDenseField2D_h_
#define _BoraMarkerDenseField2D_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

class MarkerDenseField2D : public DenseField2D<int>
{
    public:

        MarkerDenseField2D();
};

BORA_NAMESPACE_END

#endif

