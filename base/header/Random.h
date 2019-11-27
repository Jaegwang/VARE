//----------//
// Random.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2018.10.23                               //
//-------------------------------------------------------//

#ifndef _BoraRandom_h_
#define _BoraRandom_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

// float: 0.0f ~ 1.0f
BORA_FUNC_QUAL
inline float Rand( const unsigned int seed )
{
	unsigned int i = (seed^12345391)*2654435769;
	i ^= (i<<6) ^ (i>>26);
	i *= 2654435769;
	i += (i<<5) ^ (i>>12);

	return ( i / (float)(UINT_MAX) );
}

// float: min ~ max
BORA_FUNC_QUAL
inline float Rand( const unsigned int seed, const float min, const float max )
{
	unsigned int i = (seed^12345391)*2654435769;
	i ^= (i<<6) ^ (i>>26);
	i *= 2654435769;
	i += (i<<5) ^ (i>>12);

	return ( (max-min)*i / (float)(UINT_MAX) + min );
}

// int: min ~ max
BORA_FUNC_QUAL
inline int RandInt( const unsigned int seed, const int min, const int max )
{
	unsigned int i = (seed^12345391)*2654435769;
	i ^= (i<<6) ^ (i>>26);
	i *= 2654435769;
	i += (i<<5) ^ (i>>12);

	return ( i % ( max - min ) + min );
}

BORA_NAMESPACE_END

#endif

