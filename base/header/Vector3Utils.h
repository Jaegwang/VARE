//----------------//
// Vector3Utils.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.10.26                               //
//-------------------------------------------------------//

#ifndef _BoraVector3Utils_h_
#define _BoraVector3Utils_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

template <typename T> inline
Vector3<T> Center( const Vector3<T>& A, const Vector3<T>& B )
{
    return Vector3<T>( (T)(0.5*(A.x+B.x)), (T)(0.5*(A.y+B.y)), (T)(0.5*(A.z+B.z)) );
}

BORA_NAMESPACE_END

#endif

