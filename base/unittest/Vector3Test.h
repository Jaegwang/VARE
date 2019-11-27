//---------------//
// Vector3Test.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.04.19                               //
//-------------------------------------------------------//

#include <Bora.h>
using namespace std;
using namespace Bora;

BOOST_AUTO_TEST_SUITE( Vector3Suite )

BOOST_AUTO_TEST_CASE( addition )
{
    Vec3f a(120.f, 945.f, 43.f);
    Vec3f b(932.f, 192.f, 94.f);
    Vec3f c(120.f+932.f, 945.f+192.f, 43.f+94.f );

    BOOST_REQUIRE( a+b == c );
}

BOOST_AUTO_TEST_CASE( subtraction )
{
    Vec3f a(120.f, 945.f, 43.f);
    Vec3f b(932.f, 192.f, 94.f);
    Vec3f c(120.f-932.f, 945.f-192.f, 43.f-94.f );

    BOOST_REQUIRE( a-b == c );   
}

BOOST_AUTO_TEST_CASE( multiplication )
{
    Vec3f a(120.f, 945.f, 43.f);
    Vec3f b(932.f, 192.f, 94.f);
    Vec3f c(120.f-932.f, 945.f-192.f, 43.f-94.f );

    BOOST_REQUIRE( a-b == c );   
}

BOOST_AUTO_TEST_CASE( division )
{
    Vec3f a(120.f, 945.f, 43.f);
    Vec3f b(932.f, 192.f, 94.f);
    Vec3f c(120.f-932.f, 945.f-192.f, 43.f-94.f );

    BOOST_REQUIRE( a-b == c );   
}

BOOST_AUTO_TEST_CASE( dotProduct )
{
    Vec3f a(120.f, 945.f, 43.f);
    Vec3f b(932.f, 192.f, 94.f);
    float c = a.x*b.x + a.y*b.y + a.z*b.z;

    BOOST_REQUIRE( a*b == c );
}

BOOST_AUTO_TEST_CASE( casting )
{    
    glm::vec3 a(932.f, 192.f, 94.f);
    Vec3f r;
  
    r = a;
    a = r;
    BOOST_REQUIRE( r.x == a[0] && r.y == a[1] && r.z == a[2] );
    BOOST_REQUIRE( r.x == 932.f && r.y == 192.f && r.z == 94.f );
}

BOOST_AUTO_TEST_SUITE_END()

