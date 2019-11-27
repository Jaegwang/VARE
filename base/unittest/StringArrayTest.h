//-------------------//
// StringArrayTest.h //
//-------------------------------------------------------//
// author: Wano Choi @ Dexter Studios                    //
//         Jagwang Lim @ Dexter Studios                  //
// last update: 2018.01.30                               //
//-------------------------------------------------------//

#include <Bora.h>
using namespace std;
using namespace Bora;

BOOST_AUTO_TEST_SUITE( StringArraySuite )

BOOST_AUTO_TEST_CASE( stringarray_test )
{
    StringArray a( 3 );

    a.append( "aaa" );
    a.append( "bbb" );
    a.append( "ccc" );

    a[0] = "111";
    a[1] = "222";
    a[2] = "333";

    BOOST_REQUIRE( a.length() == 6 );
    BOOST_REQUIRE( a[0] == "111" );
    BOOST_REQUIRE( a[1] == "222" );
    BOOST_REQUIRE( a[2] == "333" );
    BOOST_REQUIRE( a[3] == "aaa" );
    BOOST_REQUIRE( a[4] == "bbb" );
    BOOST_REQUIRE( a[5] == "ccc" );

    StringArray b( a );

    BOOST_REQUIRE( b.length() == 6 );
    BOOST_REQUIRE( b[0] == "111" );
    BOOST_REQUIRE( b[1] == "222" );
    BOOST_REQUIRE( b[2] == "333" );
    BOOST_REQUIRE( b[3] == "aaa" );
    BOOST_REQUIRE( b[4] == "bbb" );
    BOOST_REQUIRE( b[5] == "ccc" );

    StringArray c;
    c = b;

    BOOST_REQUIRE( c.length() == 6 );
    BOOST_REQUIRE( c[0] == "111" );
    BOOST_REQUIRE( c[1] == "222" );
    BOOST_REQUIRE( c[2] == "333" );
    BOOST_REQUIRE( c[3] == "aaa" );
    BOOST_REQUIRE( c[4] == "bbb" );
    BOOST_REQUIRE( c[5] == "ccc" );

    c.erase( 3 );
    c.exclude( "111" );

    BOOST_REQUIRE( c.length() == 4 );
    BOOST_REQUIRE( c[0] == "222" );
    BOOST_REQUIRE( c[1] == "333" );
    BOOST_REQUIRE( c[2] == "bbb" );
    BOOST_REQUIRE( c[3] == "ccc" );

    BOOST_REQUIRE( a != c );

    IndexArray delList;
    delList.append( 0 );
    delList.append( 3 );

    a.remove( delList );

    BOOST_REQUIRE( a.length() == 4 );
    BOOST_REQUIRE( a[0] == "222" );
    BOOST_REQUIRE( a[1] == "333" );
    BOOST_REQUIRE( a[2] == "bbb" );
    BOOST_REQUIRE( a[3] == "ccc" );

    BOOST_REQUIRE( a == c );

    a.reverse();

    BOOST_REQUIRE( a.length() == 4 );
    BOOST_REQUIRE( a[0] == "ccc" );
    BOOST_REQUIRE( a[1] == "bbb" );
    BOOST_REQUIRE( a[2] == "333" );
    BOOST_REQUIRE( a[3] == "222" );

    a.append( c );

    BOOST_REQUIRE( a.length() == 8 );
    BOOST_REQUIRE( a[0] == "ccc" );
    BOOST_REQUIRE( a[1] == "bbb" );
    BOOST_REQUIRE( a[2] == "333" );
    BOOST_REQUIRE( a[3] == "222" );
    BOOST_REQUIRE( a[4] == "222" );
    BOOST_REQUIRE( a[5] == "333" );
    BOOST_REQUIRE( a[6] == "bbb" );
    BOOST_REQUIRE( a[7] == "ccc" );

    std::string str = a.combine("_");

    BOOST_REQUIRE( str == "ccc_bbb_333_222_222_333_bbb_ccc" );

    a.setByTokenizing( str, "_" );

    BOOST_REQUIRE( a.length() == 8 );
    BOOST_REQUIRE( a[0] == "ccc" );
    BOOST_REQUIRE( a[1] == "bbb" );
    BOOST_REQUIRE( a[2] == "333" );
    BOOST_REQUIRE( a[3] == "222" );
    BOOST_REQUIRE( a[4] == "222" );
    BOOST_REQUIRE( a[5] == "333" );
    BOOST_REQUIRE( a[6] == "bbb" );
    BOOST_REQUIRE( a[7] == "ccc" );

    a.save( "string_array_test_file" );
    a.reset();

    BOOST_REQUIRE( a.length() == 0 );

    a.load( "string_array_test_file" );

    BOOST_REQUIRE( a.length() == 8 );
    BOOST_REQUIRE( a[0] == "ccc" );
    BOOST_REQUIRE( a[1] == "bbb" );
    BOOST_REQUIRE( a[2] == "333" );
    BOOST_REQUIRE( a[3] == "222" );
    BOOST_REQUIRE( a[4] == "222" );
    BOOST_REQUIRE( a[5] == "333" );
    BOOST_REQUIRE( a[6] == "bbb" );
    BOOST_REQUIRE( a[7] == "ccc" );

    system( "rm -f string_array_test_file" );
}

BOOST_AUTO_TEST_SUITE_END()

