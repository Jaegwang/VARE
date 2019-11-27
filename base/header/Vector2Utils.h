//----------------//
// Vector2Utils.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.10.26                               //
//-------------------------------------------------------//

#ifndef _BoraVector2Utils_h_
#define _BoraVector2Utils_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

template <typename T> inline
Vector2<T> Center( const Vector2<T>& A, const Vector2<T>& B )
{
    return Vector2<T>( (T)(0.5*(A.x+B.x)), (T)(0.5*(A.y+B.y)) );
}

BORA_NAMESPACE_END

#endif

