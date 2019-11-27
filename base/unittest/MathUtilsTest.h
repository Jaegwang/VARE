//-----------------//
// MathUtilsTest.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.03.26                               //
//-------------------------------------------------------//

#include <Bora.h>
using namespace std;
using namespace Bora;

BOOST_AUTO_TEST_SUITE( MathUtilsSuite )

BOOST_AUTO_TEST_CASE( isnan )
{
    BOOST_REQUIRE( std::isnan(NAN)               == 1 );
    BOOST_REQUIRE( std::isnan(INFINITY)          == 0 );
    BOOST_REQUIRE( std::isnan(0.0)               == 0 );
    BOOST_REQUIRE( std::isnan(1.0/2.0)           == 0 );
    BOOST_REQUIRE( std::isnan(1.0/0.0)           == 0 );
    BOOST_REQUIRE( std::isnan(0.0/0.0)           == 1 );
    BOOST_REQUIRE( std::isnan(INFINITY-INFINITY) == 1 );
    BOOST_REQUIRE( std::isnan(exp(800))          == 0 );

    BOOST_REQUIRE( IsNan(NAN)               == 1 );
    BOOST_REQUIRE( IsNan(INFINITY)          == 0 );
    BOOST_REQUIRE( IsNan(0.0)               == 0 );
    BOOST_REQUIRE( IsNan(1.0/2.0)           == 0 );
    BOOST_REQUIRE( IsNan(1.0/0.0)           == 0 );
    BOOST_REQUIRE( IsNan(0.0/0.0)           == 1 );
    BOOST_REQUIRE( IsNan(INFINITY-INFINITY) == 1 );
    BOOST_REQUIRE( IsNan(exp(800))          == 0 );
}

BOOST_AUTO_TEST_CASE( isinf )
{
    BOOST_REQUIRE( std::isinf(NAN)               == 0 );
    BOOST_REQUIRE( std::isinf(INFINITY)          == 1 );
    BOOST_REQUIRE( std::isinf(0.0)               == 0 );
    BOOST_REQUIRE( std::isinf(1.0/2.0)           == 0 );
    BOOST_REQUIRE( std::isinf(1.0/0.0)           == 1 );
    BOOST_REQUIRE( std::isinf(0.0/0.0)           == 0 );
    BOOST_REQUIRE( std::isinf(INFINITY-INFINITY) == 0 );
    BOOST_REQUIRE( std::isinf(exp(800))          == 1 );

    BOOST_REQUIRE( IsInf(NAN)               == 0 );
    BOOST_REQUIRE( IsInf(INFINITY)          == 1 );
    BOOST_REQUIRE( IsInf(0.0)               == 0 );
    BOOST_REQUIRE( IsInf(1.0/2.0)           == 0 );
    BOOST_REQUIRE( IsInf(1.0/0.0)           == 1 );
    BOOST_REQUIRE( IsInf(0.0/0.0)           == 0 );
    BOOST_REQUIRE( IsInf(INFINITY-INFINITY) == 0 );
    BOOST_REQUIRE( IsInf(exp(800))          == 1 );
}

BOOST_AUTO_TEST_CASE( wrap )
{    
    BOOST_REQUIRE( Wrap( 1, 3 )  == 1 );
    BOOST_REQUIRE( Wrap( 5, 3 )  == 2 );
    BOOST_REQUIRE( Wrap( 6, 3 )  == 0 );
    BOOST_REQUIRE( Wrap( -1, 3 ) == 2 );

    BOOST_REQUIRE( AlmostSame( Wrap(  0.1f, 2.0f ), 0.1f, 1e-6f ) );
    BOOST_REQUIRE( AlmostSame( Wrap(  2.1f, 2.0f ), 0.1f, 1e-6f ) );
    BOOST_REQUIRE( AlmostSame( Wrap(  5.1f, 2.0f ), 1.1f, 1e-6f ) );
    BOOST_REQUIRE( AlmostSame( Wrap( -0.1f, 2.0f ), 1.9f, 1e-6f ) );

    BOOST_REQUIRE( AlmostSame( Wrap(  0.1, 2.0 ), 0.1, 1e-6 ) );
    BOOST_REQUIRE( AlmostSame( Wrap(  2.1, 2.0 ), 0.1, 1e-6 ) );
    BOOST_REQUIRE( AlmostSame( Wrap(  5.1, 2.0 ), 1.1, 1e-6 ) );
    BOOST_REQUIRE( AlmostSame( Wrap( -0.1, 2.0 ), 1.9, 1e-6 ) );
}

BOOST_AUTO_TEST_SUITE_END()

