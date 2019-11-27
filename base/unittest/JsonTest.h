//-------------//
// ArrayTest.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2018.03.08                               //
//-------------------------------------------------------//

#include <Bora.h>
using namespace std;
using namespace Bora;

BOOST_AUTO_TEST_SUITE( JsonSuite )

BOOST_AUTO_TEST_CASE( json_test_01 )
{
    JSONString js;
    {
        js.append( "aa", 0.0023f );
        js.append( "bb", 2322323 );
        js.append( "cc", 23123.0 );
        js.append( "dd", Vec3f( 123.9f, 12.9f, 232.8f ) );
        js.append( "ee", std::string("Hello") );
        js.append( "ff", 423.2223f );
        js.append( "gg", Vec3f( 1645.f, 234.f, 9833.f ) );
    }

    JSONString js_temp; 
    {
        js_temp.set( js.json() );
     
        float aa;
        js_temp.get( "aa", aa );

        int bb;
        js_temp.get( "bb", bb );

        double cc;
        js_temp.get( "cc", cc );

        Vec3f dd;
        js_temp.get( "dd", dd );

        std::string ee;
        js_temp.get( "ee", ee );

        float ff;
        js_temp.get( "ff", ff );

        Vec3f gg;
        js_temp.get( "gg", gg );

        BOOST_REQUIRE( aa == 0.0023f );
        BOOST_REQUIRE( bb == 2322323 );
        BOOST_REQUIRE( cc == 23123.0 );
        BOOST_REQUIRE( dd == Vec3f( 123.9f, 12.9f, 232.8f ) );
        BOOST_REQUIRE( ee == std::string("Hello") );
        BOOST_REQUIRE( ff == 423.2223f );
        BOOST_REQUIRE( gg == Vec3f( 1645.f, 234.f, 9833.f ) );
    }

    js_temp.save( "json_test_file" );
    js_temp.clear();
    js_temp.load( "json_test_file" );

    {
        js_temp.set( js.json() );
     
        float aa;
        js_temp.get( "aa", aa );

        int bb;
        js_temp.get( "bb", bb );

        double cc;
        js_temp.get( "cc", cc );

        Vec3f dd;
        js_temp.get( "dd", dd );

        std::string ee;
        js_temp.get( "ee", ee );

        float ff;
        js_temp.get( "ff", ff );

        Vec3f gg;
        js_temp.get( "gg", gg );

        BOOST_REQUIRE( aa == 0.0023f );
        BOOST_REQUIRE( bb == 2322323 );
        BOOST_REQUIRE( cc == 23123.0 );
        BOOST_REQUIRE( dd == Vec3f( 123.9f, 12.9f, 232.8f ) );
        BOOST_REQUIRE( ee == std::string("Hello") );
        BOOST_REQUIRE( ff == 423.2223f );
        BOOST_REQUIRE( gg == Vec3f( 1645.f, 234.f, 9833.f ) );
    }

    DeleteFile( "json_test_file" );
}

BOOST_AUTO_TEST_SUITE_END()

