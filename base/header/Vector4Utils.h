//----------------//
// Vector4Utils.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.10.26                               //
//-------------------------------------------------------//

#ifndef _BoraVector4Utils_h_
#define _BoraVector4Utils_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

template <typename T> inline
Vector4<T> Center( const Vector4<T>& A, const Vector4<T>& B )
{
    return Vector4<T>( (T)(0.5*(A.x+B.x)), (T)(0.5*(A.y+B.y)), (T)(0.5*(A.z+B.z)), (T)(0.5*(A.w+B.z)) );
}

BORA_NAMESPACE_END

#endif

