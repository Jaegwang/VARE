

#pragma once
#include <VARE.h>

VARE_NAMESPACE_BEGIN

#define EPSILON 1e-30f
#define INVALID_MAX 0xffffffffffffffff

// RadToDeg() and DegToRad() are better to be defined as macros rather than inline functions.
//
// ex) 
//
//     template <typename T>
//     VARE_UNIFIED inline T DegToRad( const T angle )
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
VARE_UNIFIED inline float  Sin( const float  x ) { return sinf(x); }
VARE_UNIFIED inline double Sin( const double x ) { return sin(x);  }

// cosVARE_UNIFIED ine
VARE_UNIFIED inline float  Cos( const float  x ) { return cosf(x); }
VARE_UNIFIED inline double Cos( const double x ) { return cos(x);  }

// tangent
VARE_UNIFIED inline float  Tan( const float  x ) { return tanf(x); }
VARE_UNIFIED inline double Tan( const double x ) { return tan(x);  }

// hyperbolic sVARE_UNIFIED ine
VARE_UNIFIED inline float  Sinh( const float  x ) { return sinhf(x); }
VARE_UNIFIED inline double Sinh( const double x ) { return sinh(x);  }

// hyperbolic cosVARE_UNIFIED ine
VARE_UNIFIED inline float  Cosh( const float  x ) { return coshf(x); }
VARE_UNIFIED inline double Cosh( const double x ) { return cosh(x);  }

// hyperbolic tangent
VARE_UNIFIED inline float  Tanh( const float  x ) { return tanhf(x); }
VARE_UNIFIED inline double Tanh( const double x ) { return tanh(x);  }

// hyperbolic cosecant
VARE_UNIFIED inline float  Csch( const float  x ) { return 1.f/sinhf(x); }
VARE_UNIFIED inline double Csch( const double x ) { return 1.0/sinh(x);  }

// hyperbolic secant
VARE_UNIFIED inline float  Sech( const float  x ) { return 1.f/coshf(x); }
VARE_UNIFIED inline double Sech( const double x ) { return 1.0/cosh(x);  }

// hyperblic cotangent
VARE_UNIFIED inline float  Coth( const float  x ) { return coshf(x)/sinhf(x); }
VARE_UNIFIED inline double Coth( const double x ) { return cosh(x)/sinh(x);   }

// VARE_UNIFIED inverse sine
VARE_UNIFIED inline float  ASin( const float  x ) { return asinf(x); }
VARE_UNIFIED inline double ASin( const double x ) { return asin(x);  }

// VARE_UNIFIED inverse cosine
VARE_UNIFIED inline float  ACos( const float  x ) { return acosf(x); }
VARE_UNIFIED inline double ACos( const double x ) { return acos(x);  }

// VARE_UNIFIED inverse tangent
VARE_UNIFIED inline float  ATan( const float  x ) { return atanf(x); }
VARE_UNIFIED inline double ATan( const double x ) { return atan(x);  }

// arctan(y/x)
VARE_UNIFIED inline float  ATan2( const float  x, const float  y ) { return atan2f(x,y); }
VARE_UNIFIED inline double ATan2( const double x, const double y ) { return atan2(x,y);  }

// the length of the hypotenuse of a right-angle triangle
VARE_UNIFIED inline float  Hypot( const float  x, const float  y ) { return hypotf(x,y); }
VARE_UNIFIED inline double Hypot( const double x, const double y ) { return hypot(x,y);  }

// square root
VARE_UNIFIED inline float  Sqrt( const float  x ) { return sqrtf(x); }
VARE_UNIFIED inline double Sqrt( const double x ) { return sqrt(x);  }

// exponential
VARE_UNIFIED inline float  Exp( const float  x ) { return expf(x); }
VARE_UNIFIED inline double Exp( const double x ) { return exp(x);  }

// logarithm
VARE_UNIFIED inline float  Log( const float  x ) { return logf(x); }
VARE_UNIFIED inline double Log( const double x ) { return log(x);  }

// power
VARE_UNIFIED inline float  Pow( const float  x, const float  y ) { return powf(x,y); }
VARE_UNIFIED inline double Pow( const double x, const double y ) { return pow(x,y);  }

// gamma
VARE_UNIFIED inline float  TGamma( const float  x ) { return tgammaf(x); }
VARE_UNIFIED inline double TGamma( const double x ) { return tgamma(x);  }

// floor
VARE_UNIFIED inline float  Floor( const float  x ) { return floorf(x); }
VARE_UNIFIED inline double Floor( const double x ) { return floor(x);  }

// ceil
VARE_UNIFIED inline float  Ceil( const float  x ) { return ceilf(x); }
VARE_UNIFIED inline double Ceil( const double x ) { return ceil(x);  }

// round
VARE_UNIFIED inline float  Round( const float  x ) { return roundf(x); }
VARE_UNIFIED inline double Round( const double x ) { return round(x);  }

template <typename T>
VARE_UNIFIED inline void Swap( T& a, T& b )
{
	const T c = a;
	a = b;
	b = c;
}

template <typename T>
VARE_UNIFIED inline T Abs( const T& x )
{
    return ( ( x < 0 ) ? -x : x );
}

template <typename T>
VARE_UNIFIED inline int Sign( const T& x )
{
    return ( ( x < 0 ) ? -1 : ( x > 0 ) ? 1 : 0 );
}

template <typename T>
VARE_UNIFIED inline T Power( T x, int n )
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
VARE_UNIFIED inline bool AlmostZero( const T& x, const T& eps=(T)EPSILON )
{
	return ( ( Abs(x) < eps ) ? true : false );
}

template <typename T>
VARE_UNIFIED inline bool AlmostSame( const T& a, const T& b, const T eps=(T)EPSILON )
{
	return ( ( Abs(a-b) < eps ) ? true : false );
}

template <typename T>
VARE_UNIFIED inline T Min( const T& a, const T& b )
{
	return ( (a<b) ? a : b );
}

template <typename T>
VARE_UNIFIED inline T Min( const T& a, const T& b, const T& c )
{
	return ( (a<b) ? ( (a<c) ? a : c ) : ( (b<c) ? b : c ) );
}

template <typename T>
VARE_UNIFIED inline T Min( const T& a, const T& b, const T& c, const T& d )
{
	return ( (a<b) ? ( (a<c) ? ( (a<d) ? a : d ) : ( (c<d) ? c : d ) ) : ( (b<c) ? ( (b<d) ? b : d ) : ( (c<d) ? c : d ) ) );
}

template <typename T>
VARE_UNIFIED inline T Max( const T& a, const T& b )
{
	return ( (a>b) ? a : b );
}

template <typename T>
VARE_UNIFIED inline T Max( const T& a, const T& b, const T& c )
{
	return ( (a>b) ? ( (a>c) ? a : c ) : ( (b>c) ? b : c ) );
}

template <typename T>
VARE_UNIFIED inline T Max( const T& a, const T& b, const T& c, const T& d )
{
	return ( (a>b) ? ( (a>c) ? ( (a>d) ? a : d ) : ( (c>d) ? c : d ) ) : ( (b>c) ? ( (b>d) ? b : d ) : ( (c>d) ? c : d ) ) );
}

template <typename T>
VARE_UNIFIED inline T AbsMax( const T& a, const T& b )
{
	const T A = Abs(a);
	const T B = Abs(b);

	return ( (A>B) ? A : B );
}

template <typename T>
VARE_UNIFIED inline T AbsMax( const T& a, const T& b, const T& c )
{
	const T A = Abs(a);
	const T B = Abs(b);
	const T C = Abs(c);

	return ( (A>B) ? ( (A>C) ? A : C ) : ( (B>C) ? B : C ) );
}

template <typename T>
VARE_UNIFIED inline T AbsMax( const T& a, const T& b, const T& c, const T& d )
{
	const T A = Abs(a);
	const T B = Abs(b);
	const T C = Abs(c);
	const T D = Abs(d);

	return ( (A>B) ? ( (A>C) ? ( (A>D) ? A : D ) : ( (C>D) ? C : D ) ) : ( (B>C) ? ( (B>D) ? B : D ) : ( (C>D) ? C : D ) ) );
}

template <typename T>
VARE_UNIFIED inline void GetMinMax( const T& a, const T& b, T& min, T& max )
{
	if( a < b ) { min = a; max = b; }
	else        { min = b; max = a; }
}

template <typename T>
VARE_UNIFIED inline void GetMinMax( const T& a, const T& b, const T& c, T& min, T& max )
{
	min = max = a;

	if( b < min ) { min = b; }
	if( b > max ) { max = b; }

	if( c < min ) { min = c; }
	if( c > max ) { max = c; }
}

template <typename T>
VARE_UNIFIED inline void GetMinMax( const T& a, const T& b, const T& c, const T& d, T& min, T& max )
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
VARE_UNIFIED inline T Clamp( const T x, const T low, const T high )
{
    if( x < low  ) { return low;  }
    if( x > high ) { return high; }

    return x;
}

template <typename T>
VARE_UNIFIED inline T Pow2( const T& x )
{
    return (x*x);
}

template <typename T>
VARE_UNIFIED inline T Pow3( const T& x )
{
    return (x*x*x);
}

template <typename T>
VARE_UNIFIED inline T Pow4( const T& x )
{
    return (x*x*x*x);
}

template <typename T>
VARE_UNIFIED inline T Pow5( const T& x )
{
    return (x*x*x*x*x);
}

template <typename T>
VARE_UNIFIED inline T Pow6( const T& x )
{
    return (x*x*x*x*x*x);
}

VARE_UNIFIED inline bool IsPowersOfTwo( const int n )
{
	return ( n&(n-1) ? false : true );
}

template <typename T>
VARE_UNIFIED inline T PowerOfTwo( const T exponent )
{
    return ( 1 << exponent );
}

template <typename T>
VARE_UNIFIED inline T Lerp( const T a, const T b, const float t )
{
	if( t < 0.0f ) { return a; }
	if( t > 1.0f ) { return b; }

	return ( a*(1.0f-t) + b*t );
}

template <typename T>
VARE_UNIFIED inline T SmoothStep( const T x, const T xMin=0, const T xMax=1 )
{
	if( x < xMin ) { return 0; }
	if( x > xMax ) { return 1; }

	const T t = (x-xMin) / (xMax-xMin); // Normalize x.
	return ( t*(t*(-2*t+3)) );
}

template <typename T>
VARE_UNIFIED inline T Fade( const T x, const T xMin=0, const T xMax=1 )
{
	if( x < xMin ) { return 0; }
	if( x > xMax ) { return 1; }

	const T t = (x-xMin) / (xMax-xMin);
	return ( t*t*t*(t*(t*6-15)+10) );
}

template <typename T>
VARE_UNIFIED inline T Fit( const T oldValue, const T oldMin, const T oldMax, const T newMin, const T newMax )
{
	if( oldValue < oldMin ) { return newMin; }
	if( oldValue > oldMax ) { return newMax; }

	return ( (oldValue-oldMin)*((newMax-newMin)/(oldMax-oldMin)) + newMin );
}

VARE_UNIFIED inline int Wrap( int x, int N )
{
    if( x < 0 )
    {
        x += N * ( (-x/N) + 1 );
    }

    return ( x % N );
}

VARE_UNIFIED inline float Wrap( float x, float n )
{
    return ( x - ( n * Floor( x / n ) ) );
}

VARE_UNIFIED
inline double Wrap( double x, double n )
{
    return ( x - ( n * Floor( x / n ) ) );
}

template <typename T>
VARE_UNIFIED inline void Sort( T& a, T& b, bool increasingOrder=true )
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
VARE_UNIFIED inline void Sort( T& a, T& b, T& c, bool increasingOrder=true )
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
VARE_UNIFIED inline T NumericalQuadrature( FUNC f, const T a, const T b, const int n=100 )
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
VARE_UNIFIED inline T CatRom( const T& P1, const T& P2, const T& P3, const T& P4, const float t )
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

VARE_NAMESPACE_END
