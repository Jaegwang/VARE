//----------------//
// EqnuationTest.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.11.27                               //
//-------------------------------------------------------//

#include <Bora.h>
using namespace std;
using namespace Bora;

BOOST_AUTO_TEST_SUITE( EqnuationSuite )

float          x[3];
Complex<float> y[3];

BOOST_AUTO_TEST_CASE( linear_equation_test )
{
    // 3x + 5 = 0
    {
        const float a=3, b=5;

        float x;
        SolveLinearEqn( a, b, x );

        BOOST_REQUIRE( AlmostZero( a*x+b, 0.001f ) );
    }
}

BOOST_AUTO_TEST_CASE( quadratic_equation_test )
{
    // (3x-1)(x-2) = 3x^2 + 5x + 2 = 0
    {
        const float a=3, b=5, c=2;

        // real case
        {
            const int nRoots = SolveQuadraticEqn( a, b, c, x );
            BOOST_REQUIRE( nRoots == 2 );
            BOOST_REQUIRE( AlmostZero( a*Pow2(x[0])+b*x[0]+c, 0.001f ) );
            BOOST_REQUIRE( AlmostZero( a*Pow2(x[1])+b*x[1]+c, 0.001f ) );
        }

        // complex case
        {
            const int nRoots = SolveQuadraticEqn( a, b, c, y );
            BOOST_REQUIRE( nRoots == 2 );
            BOOST_REQUIRE( AlmostZero( (a*Pow2(y[0])+b*y[0]+c).radius(), 0.001f ) );
            BOOST_REQUIRE( AlmostZero( (a*Pow2(y[1])+b*y[1]+c).radius(), 0.001f ) );
        }
    }

    // (x+1)^2 = x^2 + 2x + 1 = 0
    {
        const float a=1, b=2, c=1;

        // real case
        {
            const int nRoots = SolveQuadraticEqn( a, b, c, x );
            BOOST_REQUIRE( nRoots == 1 );
            BOOST_REQUIRE( AlmostZero( a*Pow2(x[0])+b*x[0]+c, 0.001f ) );
            BOOST_REQUIRE( AlmostZero( a*Pow2(x[1])+b*x[1]+c, 0.001f ) );
        }

        // complex case
        {
            const int nRoots = SolveQuadraticEqn( a, b, c, y );
            BOOST_REQUIRE( nRoots == 1 );
            BOOST_REQUIRE( AlmostZero( (a*Pow2(y[0])+b*y[0]+c).radius(), 0.001f ) );
            BOOST_REQUIRE( AlmostZero( (a*Pow2(y[1])+b*y[1]+c).radius(), 0.001f ) );
        }
    }

    // x^2 + 1 = 0
    {
        const float a=1, b=0, c=1;

        // real case
        {
            const int nRoots = SolveQuadraticEqn( a, b, c, x );
            BOOST_REQUIRE( nRoots == 0 );
            BOOST_REQUIRE( std::isnan(x[0]) );
            BOOST_REQUIRE( std::isnan(x[1]) );
        }

        // complex case
        {
            const int nRoots = SolveQuadraticEqn( a, b, c, y );
            BOOST_REQUIRE( nRoots == 2 );
            BOOST_REQUIRE( AlmostZero( (a*Pow2(y[0])+b*y[0]+c).radius(), 0.001f ) );
            BOOST_REQUIRE( AlmostZero( (a*Pow2(y[1])+b*y[1]+c).radius(), 0.001f ) );
        }
    }
}

BOOST_AUTO_TEST_CASE( cubic_equation_test )
{
    // (x-1)(x-1)(x-1) = x^3 -3x^2 + 3x -1
    {
        const float a=1, b=-3, c=3, d=-1;

        // real case
        {
            const int nRoots = SolveCubicEqn( a, b, c, d, x );
            BOOST_REQUIRE( nRoots == 1 );
            BOOST_REQUIRE( AlmostZero( a*Pow3(x[0])+b*Pow2(x[0])+c*x[0]+d, 0.001f ) );
            BOOST_REQUIRE( AlmostZero( a*Pow3(x[1])+b*Pow2(x[1])+c*x[1]+d, 0.001f ) );
            BOOST_REQUIRE( AlmostZero( a*Pow3(x[2])+b*Pow2(x[2])+c*x[2]+d, 0.001f ) );
        }

        // compolex case
        {
            const int nRoots = SolveCubicEqn( a, b, c, d, y );
            BOOST_REQUIRE( nRoots == 1 );
            BOOST_REQUIRE( AlmostZero( (a*Pow3(y[0])+b*Pow2(y[0])+c*y[0]+d).radius(), 0.001f ) );
            BOOST_REQUIRE( AlmostZero( (a*Pow3(y[1])+b*Pow2(y[1])+c*y[1]+d).radius(), 0.001f ) );
            BOOST_REQUIRE( AlmostZero( (a*Pow3(y[2])+b*Pow2(y[2])+c*y[2]+d).radius(), 0.001f ) );
        }
    }

    // (x-1)(x-1)(x+1) = x^3 - x^2 - x + 1
    {
        const float a=1, b=-1, c=-1, d=1;

        // real case
        {
            const int nRoots = SolveCubicEqn( a, b, c, d, x );
            BOOST_REQUIRE( nRoots == 2 );
            BOOST_REQUIRE( AlmostZero( a*Pow3(x[0])+b*Pow2(x[0])+c*x[0]+d, 0.001f ) );
            BOOST_REQUIRE( AlmostZero( a*Pow3(x[1])+b*Pow2(x[1])+c*x[1]+d, 0.001f ) );
            BOOST_REQUIRE( AlmostZero( a*Pow3(x[2])+b*Pow2(x[2])+c*x[2]+d, 0.001f ) );
        }

        // compolex case
        {
            const int nRoots = SolveCubicEqn( a, b, c, d, y );
            BOOST_REQUIRE( nRoots == 2 );
            BOOST_REQUIRE( AlmostZero( (a*Pow3(y[0])+b*Pow2(y[0])+c*y[0]+d).radius(), 0.001f ) );
            BOOST_REQUIRE( AlmostZero( (a*Pow3(y[1])+b*Pow2(y[1])+c*y[1]+d).radius(), 0.001f ) );
            BOOST_REQUIRE( AlmostZero( (a*Pow3(y[2])+b*Pow2(y[2])+c*y[2]+d).radius(), 0.001f ) );
        }
    }

    // (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
    {
        const float a=1, b=-6, c=11, d=-6;

        // real case
        {
            const int nRoots = SolveCubicEqn( a, b, c, d, x );
            BOOST_REQUIRE( nRoots == 3 );
            BOOST_REQUIRE( AlmostZero( a*Pow3(x[0])+b*Pow2(x[0])+c*x[0]+d, 0.001f ) );
            BOOST_REQUIRE( AlmostZero( a*Pow3(x[1])+b*Pow2(x[1])+c*x[1]+d, 0.001f ) );
            BOOST_REQUIRE( AlmostZero( a*Pow3(x[2])+b*Pow2(x[2])+c*x[2]+d, 0.001f ) );
        }

        // compolex case
        {
            const int nRoots = SolveCubicEqn( a, b, c, d, y );
            BOOST_REQUIRE( nRoots == 3 );
            BOOST_REQUIRE( AlmostZero( (a*Pow3(y[0])+b*Pow2(y[0])+c*y[0]+d).radius(), 0.001f ) );
            BOOST_REQUIRE( AlmostZero( (a*Pow3(y[1])+b*Pow2(y[1])+c*y[1]+d).radius(), 0.001f ) );
            BOOST_REQUIRE( AlmostZero( (a*Pow3(y[2])+b*Pow2(y[2])+c*y[2]+d).radius(), 0.001f ) );
        }
    }

    // x(x^2+1) = x^3 + x = 0
    {
        const float a=1, b=0, c=0, d=1;

        // real case
        {
            const int nRoots = SolveCubicEqn( a, b, c, d, x );
            BOOST_REQUIRE( nRoots == 1 );
            BOOST_REQUIRE( AlmostZero( a*Pow3(x[0])+b*Pow2(x[0])+c*x[0]+d, 0.001f ) );
            BOOST_REQUIRE( std::isnan(x[1]) );
            BOOST_REQUIRE( std::isnan(x[2]) );
        }

        // compolex case
        {
            const int nRoots = SolveCubicEqn( a, b, c, d, y );
            BOOST_REQUIRE( nRoots == 3 );
            BOOST_REQUIRE( AlmostZero( (a*Pow3(y[0])+b*Pow2(y[0])+c*y[0]+d).radius(), 0.001f ) );
            BOOST_REQUIRE( AlmostZero( (a*Pow3(y[1])+b*Pow2(y[1])+c*y[1]+d).radius(), 0.001f ) );
            BOOST_REQUIRE( AlmostZero( (a*Pow3(y[2])+b*Pow2(y[2])+c*y[2]+d).radius(), 0.001f ) );
        }
    }
}

// 2x + 3y + 1 = 0
// 5x -  y + 3 = 0
BOOST_AUTO_TEST_CASE( simultaneous_equation_test )
{
    const float a =  2;
    const float b =  3;
    const float c =  1;

    const float p =  5;
    const float q = -1;
    const float r =  3;

    float x=0, y=0;

    SolveSimultaneousEqn( a,b,c, p,q,r, x,y );

    BOOST_REQUIRE( AlmostZero( a*x+b*y+c, 0.001f ) );
    BOOST_REQUIRE( AlmostZero( p*x+q*y+r, 0.001f ) );
}

BOOST_AUTO_TEST_SUITE_END()

