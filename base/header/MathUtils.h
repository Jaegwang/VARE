//-------------//
// MathUtils.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2018.12.20                               //
//-------------------------------------------------------//

#ifndef _BoraMathUtils_h_
#define _BoraMathUtils_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

// RadToDeg() and DegToRad() are better to be defined as macros rather than inline functions.
//
// ex) 
//
//     template <typename T>
//     BORA_FUNC_QUAL inline T DegToRad( const T angle )
//     {
//         return ( angle * 0.0174532925199432954743716805978692718782f );
//     }
//
//     float a = DegToRad( 30 );
//     
//     a = 0 (not 0.52)

#define RadToDeg(x) ((x)*57.295779513082322864647721871733665466309f)
#define DegToRad(x) ((x)*0.0174532925199432954743716805978692718782f)

#define IsNan(x)  ((!((x)<0))&&(!((x)>=0)))
#define IsInf(x)  (!IsNan(x)&&IsNan(x-x))

#define IsOdd(x)  ((x)&1)
#define IsEven(x) (!((x)&1))

// sine
BORA_FUNC_QUAL inline float  Sin( const float  x ) { return sinf(x); }
BORA_FUNC_QUAL inline double Sin( const double x ) { return sin(x);  }

// cosBORA_FUNC_QUAL ine
BORA_FUNC_QUAL inline float  Cos( const float  x ) { return cosf(x); }
BORA_FUNC_QUAL inline double Cos( const double x ) { return cos(x);  }

// tangent
BORA_FUNC_QUAL inline float  Tan( const float  x ) { return tanf(x); }
BORA_FUNC_QUAL inline double Tan( const double x ) { return tan(x);  }

// hyperbolic sBORA_FUNC_QUAL ine
BORA_FUNC_QUAL inline float  Sinh( const float  x ) { return sinhf(x); }
BORA_FUNC_QUAL inline double Sinh( const double x ) { return sinh(x);  }

// hyperbolic cosBORA_FUNC_QUAL ine
BORA_FUNC_QUAL inline float  Cosh( const float  x ) { return coshf(x); }
BORA_FUNC_QUAL inline double Cosh( const double x ) { return cosh(x);  }

// hyperbolic tangent
BORA_FUNC_QUAL inline float  Tanh( const float  x ) { return tanhf(x); }
BORA_FUNC_QUAL inline double Tanh( const double x ) { return tanh(x);  }

// hyperbolic cosecant
BORA_FUNC_QUAL inline float  Csch( const float  x ) { return 1.f/sinhf(x); }
BORA_FUNC_QUAL inline double Csch( const double x ) { return 1.0/sinh(x);  }

// hyperbolic secant
BORA_FUNC_QUAL inline float  Sech( const float  x ) { return 1.f/coshf(x); }
BORA_FUNC_QUAL inline double Sech( const double x ) { return 1.0/cosh(x);  }

// hyperblic cotangent
BORA_FUNC_QUAL inline float  Coth( const float  x ) { return coshf(x)/sinhf(x); }
BORA_FUNC_QUAL inline double Coth( const double x ) { return cosh(x)/sinh(x);   }

// BORA_FUNC_QUAL inverse sine
BORA_FUNC_QUAL inline float  ASin( const float  x ) { return asinf(x); }
BORA_FUNC_QUAL inline double ASin( const double x ) { return asin(x);  }

// BORA_FUNC_QUAL inverse cosine
BORA_FUNC_QUAL inline float  ACos( const float  x ) { return acosf(x); }
BORA_FUNC_QUAL inline double ACos( const double x ) { return acos(x);  }

// BORA_FUNC_QUAL inverse tangent
BORA_FUNC_QUAL inline float  ATan( const float  x ) { return atanf(x); }
BORA_FUNC_QUAL inline double ATan( const double x ) { return atan(x);  }

// arctan(y/x)
BORA_FUNC_QUAL inline float  ATan2( const float  x, const float  y ) { return atan2f(x,y); }
BORA_FUNC_QUAL inline double ATan2( const double x, const double y ) { return atan2(x,y);  }

// the length of the hypotenuse of a right-angle triangle
BORA_FUNC_QUAL inline float  Hypot( const float  x, const float  y ) { return hypotf(x,y); }
BORA_FUNC_QUAL inline double Hypot( const double x, const double y ) { return hypot(x,y);  }

// square root
BORA_FUNC_QUAL inline float  Sqrt( const float  x ) { return sqrtf(x); }
BORA_FUNC_QUAL inline double Sqrt( const double x ) { return sqrt(x);  }

// exponential
BORA_FUNC_QUAL inline float  Exp( const float  x ) { return expf(x); }
BORA_FUNC_QUAL inline double Exp( const double x ) { return exp(x);  }

// logarithm
BORA_FUNC_QUAL inline float  Log( const float  x ) { return logf(x); }
BORA_FUNC_QUAL inline double Log( const double x ) { return log(x);  }

// power
BORA_FUNC_QUAL inline float  Pow( const float  x, const float  y ) { return powf(x,y); }
BORA_FUNC_QUAL inline double Pow( const double x, const double y ) { return pow(x,y);  }

// gamma
BORA_FUNC_QUAL inline float  TGamma( const float  x ) { return tgammaf(x); }
BORA_FUNC_QUAL inline double TGamma( const double x ) { return tgamma(x);  }

// floor
BORA_FUNC_QUAL inline float  Floor( const float  x ) { return floorf(x); }
BORA_FUNC_QUAL inline double Floor( const double x ) { return floor(x);  }

// ceil
BORA_FUNC_QUAL inline float  Ceil( const float  x ) { return ceilf(x); }
BORA_FUNC_QUAL inline double Ceil( const double x ) { return ceil(x);  }

// round
BORA_FUNC_QUAL inline float  Round( const float  x ) { return roundf(x); }
BORA_FUNC_QUAL inline double Round( const double x ) { return round(x);  }

template <typename T>
BORA_FUNC_QUAL
inline T Abs( const T& x )
{
    return ( ( x < 0 ) ? -x : x );
}

template <typename T>
BORA_FUNC_QUAL
inline int Sign( const T& x )
{
    return ( ( x < 0 ) ? -1 : ( x > 0 ) ? 1 : 0 );
}

template <typename T>
BORA_FUNC_QUAL
inline T Power( T x, int n )
{
    T ans = 1;

    if( n < 0 )
    {
        n = -n;
        x = T(1) / x;
    }

    while( n-- ) { ans *= x; }

    return ans;
}

template <typename T>
BORA_FUNC_QUAL
inline bool AlmostZero( const T& x, const T& eps=(T)EPSILON )
{
	return ( ( Abs(x) < eps ) ? true : false );
}

template <typename T>
BORA_FUNC_QUAL
inline bool AlmostSame( const T& a, const T& b, const T eps=(T)EPSILON )
{
	return ( ( Abs(a-b) < eps ) ? true : false );
}

template <typename T>
BORA_FUNC_QUAL
inline T Min( const T& a, const T& b )
{
	return ( (a<b) ? a : b );
}

template <typename T>
BORA_FUNC_QUAL
inline T Min( const T& a, const T& b, const T& c )
{
	return ( (a<b) ? ( (a<c) ? a : c ) : ( (b<c) ? b : c ) );
}

template <typename T>
BORA_FUNC_QUAL
inline T Min( const T& a, const T& b, const T& c, const T& d )
{
	return ( (a<b) ? ( (a<c) ? ( (a<d) ? a : d ) : ( (c<d) ? c : d ) ) : ( (b<c) ? ( (b<d) ? b : d ) : ( (c<d) ? c : d ) ) );
}

template <typename T>
BORA_FUNC_QUAL
inline T Max( const T& a, const T& b )
{
	return ( (a>b) ? a : b );
}

template <typename T>
BORA_FUNC_QUAL
inline T Max( const T& a, const T& b, const T& c )
{
	return ( (a>b) ? ( (a>c) ? a : c ) : ( (b>c) ? b : c ) );
}

template <typename T>
BORA_FUNC_QUAL
inline T Max( const T& a, const T& b, const T& c, const T& d )
{
	return ( (a>b) ? ( (a>c) ? ( (a>d) ? a : d ) : ( (c>d) ? c : d ) ) : ( (b>c) ? ( (b>d) ? b : d ) : ( (c>d) ? c : d ) ) );
}

template <typename T>
BORA_FUNC_QUAL
inline T AbsMax( const T& a, const T& b )
{
	const T A = Abs(a);
	const T B = Abs(b);

	return ( (A>B) ? A : B );
}

template <typename T>
BORA_FUNC_QUAL
inline T AbsMax( const T& a, const T& b, const T& c )
{
	const T A = Abs(a);
	const T B = Abs(b);
	const T C = Abs(c);

	return ( (A>B) ? ( (A>C) ? A : C ) : ( (B>C) ? B : C ) );
}

template <typename T>
BORA_FUNC_QUAL
inline T AbsMax( const T& a, const T& b, const T& c, const T& d )
{
	const T A = Abs(a);
	const T B = Abs(b);
	const T C = Abs(c);
	const T D = Abs(d);

	return ( (A>B) ? ( (A>C) ? ( (A>D) ? A : D ) : ( (C>D) ? C : D ) ) : ( (B>C) ? ( (B>D) ? B : D ) : ( (C>D) ? C : D ) ) );
}

template <typename T>
BORA_FUNC_QUAL
inline void GetMinMax( const T& a, const T& b, T& min, T& max )
{
	if( a < b ) { min = a; max = b; }
	else        { min = b; max = a; }
}

template <typename T>
BORA_FUNC_QUAL
inline void GetMinMax( const T& a, const T& b, const T& c, T& min, T& max )
{
	min = max = a;

	if( b < min ) { min = b; }
	if( b > max ) { max = b; }

	if( c < min ) { min = c; }
	if( c > max ) { max = c; }
}

template <typename T>
BORA_FUNC_QUAL
inline void GetMinMax( const T& a, const T& b, const T& c, const T& d, T& min, T& max )
{
	min = max = a;

	if( b < min ) { min = b; }
	if( b > max ) { max = b; }

	if( c < min ) { min = c; }
	if( c > max ) { max = c; }

	if( d < min ) { min = d; }
	if( d > max ) { max = d; }
}

template <typename T>
BORA_FUNC_QUAL
inline T Clamp( const T x, const T low, const T high )
{
    if( x < low  ) { return low;  }
    if( x > high ) { return high; }

    return x;
}

template <typename T>
BORA_FUNC_QUAL
inline T Pow2( const T& x )
{
    return (x*x);
}

template <typename T>
BORA_FUNC_QUAL
inline T Pow3( const T& x )
{
    return (x*x*x);
}

template <typename T>
BORA_FUNC_QUAL
inline T Pow4( const T& x )
{
    return (x*x*x*x);
}

template <typename T>
BORA_FUNC_QUAL
inline T Pow5( const T& x )
{
    return (x*x*x*x*x);
}

template <typename T>
BORA_FUNC_QUAL
inline T Pow6( const T& x )
{
    return (x*x*x*x*x*x);
}

BORA_FUNC_QUAL
inline bool IsPowersOfTwo( const int n )
{
	return ( n&(n-1) ? false : true );
}

template <typename T>
BORA_FUNC_QUAL
inline T PowerOfTwo( const T exponent )
{
    return ( 1 << exponent );
}

template <typename T>
BORA_FUNC_QUAL
inline T Lerp( const T a, const T b, const float t )
{
	if( t < 0.0f ) { return a; }
	if( t > 1.0f ) { return b; }

	return ( a*(1.0f-t) + b*t );
}

template <typename T>
BORA_FUNC_QUAL
inline T SmoothStep( const T x, const T xMin=0, const T xMax=1 )
{
	if( x < xMin ) { return 0; }
	if( x > xMax ) { return 1; }

	const T t = (x-xMin) / (xMax-xMin); // Normalize x.
	return ( t*(t*(-2*t+3)) );
}

template <typename T>
BORA_FUNC_QUAL
inline T Fade( const T x, const T xMin=0, const T xMax=1 )
{
	if( x < xMin ) { return 0; }
	if( x > xMax ) { return 1; }

	const T t = (x-xMin) / (xMax-xMin);
	return ( t*t*t*(t*(t*6-15)+10) );
}

template <typename T>
BORA_FUNC_QUAL
inline T Fit( const T oldValue, const T oldMin, const T oldMax, const T newMin, const T newMax )
{
	if( oldValue < oldMin ) { return newMin; }
	if( oldValue > oldMax ) { return newMax; }

	return ( (oldValue-oldMin)*((newMax-newMin)/(oldMax-oldMin)) + newMin );
}

BORA_FUNC_QUAL
inline int Wrap( int x, int N )
{
    if( x < 0 )
    {
        x += N * ( (-x/N) + 1 );
    }

    return ( x % N );
}

BORA_FUNC_QUAL
inline float Wrap( float x, float n )
{
    return ( x - ( n * Floor( x / n ) ) );
}

BORA_FUNC_QUAL
inline double Wrap( double x, double n )
{
    return ( x - ( n * Floor( x / n ) ) );
}

template <typename T>
BORA_FUNC_QUAL
inline void Sort( T& a, T& b, bool increasingOrder=true )
{
    if( increasingOrder )
    {
        if( a > b ) { Swap( a, b ); }
    }
    else
    {
        if( a < b ) { Swap( a, b ); }
    }
}

template <typename T>
BORA_FUNC_QUAL
inline void Sort( T& a, T& b, T& c, bool increasingOrder=true )
{
    if( increasingOrder )
    {
        if( a > b ) { Swap( a, b ); }
        if( a > c ) { Swap( a, c ); }
        if( b > c ) { Swap( b, c ); }
    }
    else
    {
        if( a < b ) { Swap( a, b ); }
        if( a < c ) { Swap( a, c ); }
        if( b < c ) { Swap( b, c ); }
    }
}

template <typename FUNC, typename T>
BORA_FUNC_QUAL
inline T NumericalQuadrature( FUNC f, const T a, const T b, const int n=100 )
{
    const T dx = ( b - a ) / (T)n;

    T sum = 0;

    for( int i=1; i<n; ++i )
    {
        const T x = a + i*dx;
        sum += f( x );
    }

    return ( dx * ( (T)0.5*f(a) + sum + (T)0.5*f(b) ) );
}

template <class T>
BORA_FUNC_QUAL
inline T CatRom( const T& P1, const T& P2, const T& P3, const T& P4, const float t )
{
    if( t <   EPSILON ) { return P2; }
    if( t > 1-EPSILON ) { return P3; }

    const float tt = t*t;

    const float s1 = t * (  0.5f*tt -      t + 0.5f );
    const float s2 = t * (  1.5f*tt - 2.5f*t        ) + 1.0f;
    const float s3 = t * ( -1.5f*tt + 2.0f*t + 0.5f );
    const float s4 = 0.5f * tt * ( 1.0f - t );

    return ( (s1+s2)*Lerp(-P1,P2,s2/(s1+s2)) - (s3+s4)*Lerp(-P3,P4,s4/(s3+s4)) );
}

BORA_NAMESPACE_END

#endif

