//-----------//
// Solvers.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.11.28                               //
//-------------------------------------------------------//

#ifndef _BoraSolvers_h_
#define _BoraSolvers_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

// Linear equation
// ax + b = 0
template <typename T>
BORA_FUNC_QUAL
inline void SolveLinearEqn( const T a, const T b, T& x )
{
    if( a == 0 )
    {
        x = NAN;
    }

    x = (T)( -(double)b / (double)a );
}

// Quadratic equation for real roots
// ax^2 + bx + c = 0
template <typename T>
BORA_FUNC_QUAL
int SolveQuadraticEqn( const T a, const T b, const T c, T x[2] )
{
    if( AlmostZero(a) ) // linear case
    {
        SolveLinearEqn( b, c, x[0] );
        x[1] = x[0];

        return 1;
    }

    const double D = b*b - 4*a*c;
    const double _2a = 1 / (2*a);

    if( D==0 ) // one real root
    {
        x[0] = x[1] = (T)( -b * _2a );

        return 1;
    }
    else // two roots
    {
        if( D>0 ) // two real roots
        {
            const double s = Sqrt(D);

            x[0] = (T)( ( -b - s ) * _2a );
            x[1] = (T)( ( -b + s ) * _2a );

            return 2;
        }
        else // D<0: no roots
        {
            x[0] = x[1] = NAN;

            return 0;
        }
    }
}

// Quadratic equation for complex roots
// ax^2 + bx + c = 0
template <typename T>
BORA_FUNC_QUAL
int SolveQuadraticEqn( const T a, const T b, const T c, Complex<T> x[2] )
{
    if( AlmostZero(a) ) // linear case
    {
        SolveLinearEqn( b, c, x[0].r );
        x[1] = x[0];

        return 1;
    }

    const double D = b*b - 4*a*c;
    const double _2a = 1 / (2*a);

    if( D==0 ) // one real root
    {
        x[0] = x[1] = (T)( -b * _2a );

        return 1;
    }
    else // two roots
    {
        if( D>0 ) // two real roots
        {
            const double s = Sqrt(D);

            x[0] = (T)( ( -b - s ) * _2a );
            x[1] = (T)( ( -b + s ) * _2a );
        }
        else // D<0: two complex roots
        {
            x[0].r = (T)( -b * _2a );
            x[0].i = (T)( Sqrt( Abs(D) ) * _2a );

            x[1] = x[0].conjugated();
        }

        return 2;
    }
}

// Cubic equation for real roots
// ax^3 + bx^2 + cx + d = 0
template <typename T>
BORA_FUNC_QUAL
int SolveCubicEqn( const T a, const T b, const T c, const T d, T x[3] )
{
    if( a==0 && b==0 ) // linear case
    {
        SolveLinearEqn( c,d, x[0] );
        x[1] = x[2] = x[0];
        return 1;
    }

    if( a==0 ) // quadratic case
    {
        SolveQuadraticEqn( b,c,d, &x[0] );
        x[2] = x[0];
        return 2;
    }

    const double f = ((3.0*c/a)-((b*b)/(a*a)))/3.0;
    const double g = (((2.0*b*b*b)/(a*a*a))-((9.0*b*c)/(a*a))+(27.0*d/a))/27.0;
    const double h = ((g*g)/4.0+(f*f*f)/27.0);

    if( f==0 && g==0 && h==0 ) // all three roots are real and equal
    {
        if( d/a >= 0 )
        {
            x[0] = x[1] = x[2] = (T)( -Pow(d/a,1/(T)3) );
            return 1;
        }
        else
        {
            x[0] = x[1] = x[2] = (T)( Pow(-d/a,1/(T)3) );
            return 1;
        }
    }
    else if( h<=0 ) // all three roots are real
    {
        const double i = Sqrt(((g*g)/4.0)-h);
        const double j = Pow(i,1/3.0);
        const double k = ACos(-g/(2.0*i));
        const double L = -j;
        const double M = Cos(k/3.0);
        const double N = Sqrt(3.0)*Sin(k/3.0);
        const double P = -b/(3.0*a);

        x[0] = (T)( 2.0*j*Cos(k/3.0)-(b/(3.0*a)) );
        x[1] = (T)( L*(M+N)+P );
        x[2] = (T)( L*(M-N)+P );

        if( AlmostSame( x[0],x[1] ) ) { return 2; }
        if( AlmostSame( x[1],x[2] ) ) { return 2; }
        if( AlmostSame( x[2],x[0] ) ) { return 2; }

        return 3;
    }
    else if( h>0 ) // one real root and two complex roots
    {
        const double u = -(g/2.0)+Sqrt(h);
        const double U = Sign(u)*Pow(Abs(u),1/3.0);
        const double v = -(g/2.0)-Sqrt(h);
        const double V = Sign(v)*Pow(Abs(v),1/3.0);

        x[0] = (T)( (U+V)-(b/(3.0*a)) );
        x[1] = NAN;
        x[2] = NAN;

        return 1;
    }

    return 0;
}

// Cubic equation for complex roots
// ax^3 + bx^2 + cx + d = 0
template <typename T>
BORA_FUNC_QUAL
int SolveCubicEqn( const T a, const T b, const T c, const T d, Complex<T> x[3] )
{
    if( a==0 && b==0 ) // linear case
    {
        SolveLinearEqn( c,d, x[0].r );
        x[1] = x[2] = x[0];
        return 1;
    }

    if( a==0 ) // quadratic case
    {
        SolveQuadraticEqn( b,c,d, &x[0] );
        x[2] = x[0];
        return 2;
    }

    const double f = ((3.0*c/a)-((b*b)/(a*a)))/3.0;
    const double g = (((2.0*b*b*b)/(a*a*a))-((9.0*b*c)/(a*a))+(27.0*d/a))/27.0;
    const double h = ((g*g)/4.0+(f*f*f)/27.0);

    if( f==0 && g==0 && h==0 ) // all three roots are real and equal
    {
        if( d/a >= 0 )
        {
            x[0] = x[1] = x[2] = (T)( -Pow(d/a,1/(T)3) );
            return 1;
        }
        else
        {
            x[0] = x[1] = x[2] = (T)( Pow(-d/a,1/(T)3) );
            return 1;
        }
    }
    else if( h <= 0 ) // all three roots are real
    {
        const double i = Sqrt(((g*g)/4.0)-h);
        const double j = Pow(i,1/3.0);
        const double k = ACos(-g/(2.0*i));
        const double L = -j;
        const double M = Cos(k/3.0);
        const double N = Sqrt(3.0)*Sin(k/3.0);
        const double P = -b/(3.0*a);

        x[0] = (T)( 2.0*j*Cos(k/3.0)-(b/(3.0*a)) );
        x[1] = (T)( L*(M+N)+P );
        x[2] = (T)( L*(M-N)+P );

        if( AlmostSame( x[0].r,x[1].r ) ) { return 2; }
        if( AlmostSame( x[1].r,x[2].r ) ) { return 2; }
        if( AlmostSame( x[2].r,x[0].r ) ) { return 2; }

        return 3;
    }
    else if( h > 0 ) // one real root and two complex roots
    {
        const double u = -(g/2.0)+Sqrt(h);
        const double U = Sign(u)*Pow(Abs(u),1/3.0);
        const double v = -(g/2.0)-Sqrt(h);
        const double V = Sign(v)*Pow(Abs(v),1/3.0);

        x[0] = (T)( (U+V)-(b/(3.0*a)) );
        x[1] = Complex<T>( (T)( -(U+V)/2.0-(b/(3.0*a)) ), (T)( (U-V)*Sqrt(3.0)*0.5 ) );
        x[2] = Complex<T>( (T)( -(U+V)/2.0-(b/(3.0*a)) ), (T)(-(U-V)*Sqrt(3.0)*0.5 ) );

        return 3;
    }

    return 0;
}

// Simulataneous equation
// ax + by + c = 0
// px + qy + r = 0
template <typename T>
BORA_FUNC_QUAL
void SolveSimultaneousEqn( const T a, const T b, const T c, const T p, const T q, const T r, T& x, T& y )
{
    const double d = a*q - b*p;

    if( d == 0 )
    {
        x = y = NAN;
        return;
    }

    x = (T)( ( b*r - c*q ) / d );

    if( q == 0 )
    {
        y = NAN;
        return;
    }

    y = -1/q * ( p*x + r );
}

BORA_NAMESPACE_END

#endif

