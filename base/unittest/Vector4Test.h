//---------------//
// Vector4Test.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2017.10.26                               //
//-------------------------------------------------------//

#include <Bora.h>
using namespace std;
using namespace Bora;

BOOST_AUTO_TEST_SUITE( Vector4Suite )

BOOST_AUTO_TEST_CASE( addition )
{
    Vec4f a(120.f, 945.f, 43.f, 8723.f );
    Vec4f b(932.f, 192.f, 94.f, 1123.f );
    Vec4f c(120.f+932.f, 945.f+192.f, 43.f+94.f, 8723.f+1123.f );

    BOOST_REQUIRE( a+b == c );
}

BOOST_AUTO_TEST_CASE( subtraction )
{
    Vec4f a(120.f, 945.f, 43.f, 8723.f );
    Vec4f b(932.f, 192.f, 94.f, 1123.f );
    Vec4f c(120.f-932.f, 945.f-192.f, 43.f-94.f, 8723.f-1123.f );

    BOOST_REQUIRE( a-b == c );   
}

BOOST_AUTO_TEST_SUITE_END();

