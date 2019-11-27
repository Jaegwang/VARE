//-------------//
// MacroTest.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2017.11.28                               //
//-------------------------------------------------------//

#include <Bora.h>
using namespace std;
using namespace Bora;

BOOST_AUTO_TEST_SUITE( MacroTest )

BOOST_AUTO_TEST_CASE( comp )
{
    float a = 10.f;
    float b = 20.f;
    float c = 30.f;

    float mmin3 = Min( c, b, a );
    float mmax3 = Max( c, b, a );

    float mmin2 = Min( c, b );
    float mmax2 = Max( c, b );

    BOOST_REQUIRE( mmin2 == b );
    BOOST_REQUIRE( mmax2 == c );

    BOOST_REQUIRE( mmin3 == a );
    BOOST_REQUIRE( mmax3 == c );
}

BOOST_AUTO_TEST_SUITE_END()

