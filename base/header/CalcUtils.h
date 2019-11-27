//-------------//
// CalcUtils.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.03.26                               //
//-------------------------------------------------------//

#ifndef _BoraCalcUtils_h_
#define _BoraCalcUtils_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

// A, B: non-zero, unit length (normalized)
// atan() is more accurate than acos() for small angles
template <typename T>
BORA_FUNC_QUAL
inline T Angle( const Vector3<T>& A, const Vector3<T>& B )
{
	return ATan2( (A^B).length(), A*B );
}

template <typename T>
BORA_FUNC_QUAL
inline Vector3<T> RotateVector( const Vector3<T>& v, const Vector3<T>& unitAxis, const T angleInRadians )
{
    Vector3<T> V( v );

    const T c = Cos( angleInRadians );
    const T s = Sin( angleInRadians );

    const Vector3<T> cross( unitAxis ^ V );
    const T dot = unitAxis * V;

    const T alpha = (1-c) * dot;

    V.x = c*V.x + s*cross.x + alpha*unitAxis.x;
    V.y = c*V.y + s*cross.y + alpha*unitAxis.y;
    V.z = c*V.z + s*cross.z + alpha*unitAxis.z;

    return V;
}

template <typename T>
BORA_FUNC_QUAL
inline Vector3<T> RotatePoint( const Vector3<T>& p, const Vector3<T>& unitAxis, const T angleInRadians, const Vector3<T>& pivot )
{
    Vector3<T> P( p - pivot );

    RotateVector( P, unitAxis, angleInRadians );

    return ( P += pivot );
}

BORA_NAMESPACE_END

#endif

