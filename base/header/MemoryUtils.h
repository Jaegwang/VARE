//---------------//
// MemoryUtils.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.04.24                               //
//-------------------------------------------------------//

#ifndef _BoraMemoryUtils_h_
#define _BoraMemoryUtils_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

template <class T>
BORA_FUNC_QUAL
inline void Swap( T& a, T& b )
{
    T c = a;

    a = b;
    b = c;    
}

template <class T>
inline void SwitchEndian( T& value )
{ 
    const T tmp = value;

    char* src = (char*)&tmp;
    char* dst = (char*)&value;

    const int size = sizeof(T);

    for( int i=0; i<size; ++i )
    {
        dst[i] = src[size-i-1];
    }
}

template <typename T>
inline bool GetBit( const T x, int N )
{
    return ( ( x & (1<<N) ) >> N );
}

template <typename T>
inline void PrintBits( const T x, int N )
{
    switch( N )
    {
        case  4: { COUT << std::bitset< 4>(x) << ENDL; break; }
        case  8: { COUT << std::bitset< 8>(x) << ENDL; break; }
        case 16: { COUT << std::bitset<16>(x) << ENDL; break; }
        case 32: { COUT << std::bitset<32>(x) << ENDL; break; }
        case 64: { COUT << std::bitset<64>(x) << ENDL; break; }
        default: { break; }
    }
}

inline uint32_t HowManyOnesInBits( uint32_t x )
{
    // count number of ones
    x = (x & 0x55555555) + ((x >> 1) & 0x55555555); // add pairs of bits
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333); // add bit pairs
    x = (x & 0x0f0f0f0f) + ((x >> 4) & 0x0f0f0f0f); // add nybbles
    x += (x >> 8);                                  // add bytes
    x += (x >> 16);                                 // add words

    return(x & 0xff);
}

template <typename T>
inline void SetBit( T& x, int N )
{
    x |= (1<<N);
}

template <typename T>
inline T ClearBit( T& x, int N )
{
    x &= ~(1<<N);
}

template <class T>
inline void Zeroize( T& v )
{
    char* x = (char*)&v;
    const size_t size = sizeof(T);

    for( size_t i=0; i<size; ++i )
    {
        x[i]=0;
    }
}

inline std::string MemorySize( const size_t bytes )
{
    std::stringstream ss;

    if( bytes < 1024 )
    {
        ss << bytes;
        return ( ss.str() + " bytes" );
    }

    if( bytes < (1024*1024) )
    {
        ss << ( bytes / 1024.0 );
        return ( ss.str() + " KB" );
    }

    if( bytes < (1024*1024*1024) )
    {
        ss << ( bytes / (1024.0*1024.0) );
        return ( ss.str() + " MB" );
    }

    ss << ( bytes / (1024.0*1024.0*1024.0) );
    return ( ss.str() + " GB" );
}

BORA_NAMESPACE_END

#endif

