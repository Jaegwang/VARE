//-------------//
// ArrayTest.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2018.03.20                               //
//-------------------------------------------------------//

#include <Bora.h>
using namespace std;
using namespace Bora;

BOOST_AUTO_TEST_SUITE( ArraySuite )

BOOST_AUTO_TEST_CASE( array_test_01 )
{
    IntArray a;

    BOOST_REQUIRE( a.size() == 0 );
    BOOST_REQUIRE( a.capacity() == 0 );

    a.initialize( 10 );

    BOOST_REQUIRE( a.size() == 10 );
    BOOST_REQUIRE( a.capacity() == 10 );

    a.reserve( 20 );

    BOOST_REQUIRE( a.size() == 10 );
    BOOST_REQUIRE( a.capacity() == 20 );

    a.resize( 5 );

    BOOST_REQUIRE( a.size() == 5 );
    BOOST_REQUIRE( a.capacity() == 5 );

    a.shrink();

    BOOST_REQUIRE( a.size() == 5 );
    BOOST_REQUIRE( a.capacity() == 5 );

    a.resize( 15 );

    BOOST_REQUIRE( a.size() == 15 );
    BOOST_REQUIRE( a.capacity() == 15 );

}

BOOST_AUTO_TEST_CASE( array_test_02 )
{
    IntArray a;

    BOOST_REQUIRE( a.size() == 0 );
    BOOST_REQUIRE( a.capacity() == 0 );

    a.resize( 3 );

    BOOST_REQUIRE( a.size() == 3 );
    BOOST_REQUIRE( a.capacity() == 3 );

    a.finalize();

    BOOST_REQUIRE( a.size() == 0 );
    BOOST_REQUIRE( a.capacity() == 0 );

    a.resize( 10 );

    BOOST_REQUIRE( a.size() == 10 );
    BOOST_REQUIRE( a.capacity() == 10 );

    a.setOffsetValues( 3, 123 );

    for( size_t i=0; i<a.size(); ++i )
    {
        BOOST_REQUIRE( a[i] == (3+i*123) );
    }

    IntArray b;

    // Don't use b = a
    b.copyFrom( a );

    BOOST_REQUIRE( a == b );

    b.reserve( 100 );

    BOOST_REQUIRE( b.size() == 10  );
    BOOST_REQUIRE( b.capacity() == 100 );

    for( size_t i=0; i<b.size(); ++i )
    {
        BOOST_REQUIRE( b[i] == (3+i*123) );
    }

    b.shrink();

    BOOST_REQUIRE( b.size()     == 10 );
    BOOST_REQUIRE( b.capacity() == 10 );

    BOOST_REQUIRE( b.minValue() == 3    );
    BOOST_REQUIRE( b.maxValue() == 1110 );
}

BOOST_AUTO_TEST_CASE( array_test_03 )
{
    IntArray a, b;

    a.resize(3);
    a.setOffsetValues(0,1);
    a.append(3);

    BOOST_REQUIRE( a[0] == 0 );
    BOOST_REQUIRE( a[1] == 1 );
    BOOST_REQUIRE( a[2] == 2 );
    BOOST_REQUIRE( a[3] == 3 );

    b.resize(3);
    b.setOffsetValues(4,1);
    b.append(7);

    BOOST_REQUIRE( b[0] == 4 );
    BOOST_REQUIRE( b[1] == 5 );
    BOOST_REQUIRE( b[2] == 6 );
    BOOST_REQUIRE( b[3] == 7 );

    a.append( b );

    BOOST_REQUIRE( a.size() == 8 );
    BOOST_REQUIRE( a.capacity() == 8 );

    BOOST_REQUIRE( a[4] == 4 );
    BOOST_REQUIRE( a[5] == 5 );
    BOOST_REQUIRE( a[6] == 6 );
    BOOST_REQUIRE( a[7] == 7 );

    IndexArray delList;
    delList.append( 1 );
    delList.append( 3 );
    delList.append( 5 );
    delList.append( 7 );
/*
    a.eliminate( delList );

    BOOST_REQUIRE( a.size() == 4 );
    BOOST_REQUIRE( a[0] == 0 );
    BOOST_REQUIRE( a[1] == 2 );
    BOOST_REQUIRE( a[2] == 4 );
    BOOST_REQUIRE( a[3] == 6 );

    // Don't use b = a
    b.copyFrom( a );

    a.erase( 1 );

    BOOST_REQUIRE( a.size() == 3 );
    BOOST_REQUIRE( a[0] == 0 );
    BOOST_REQUIRE( a[1] == 4 );
    BOOST_REQUIRE( a[2] == 6 );

    a.insert( 1, 2 );

    BOOST_REQUIRE( a.size() == 4 );
    BOOST_REQUIRE( a[0] == 0 );
    BOOST_REQUIRE( a[1] == 2 );
    BOOST_REQUIRE( a[2] == 4 );
    BOOST_REQUIRE( a[3] == 6 );

    BOOST_REQUIRE( a == b );
*/    
}

BOOST_AUTO_TEST_CASE( array_test_04 )
{
    IntArray a;
    a.append( 1 );
    a.append( 1 );
    a.append( 1 );
    a.append( 2 );
    a.append( 2 );
    a.append( 3 );
    a.append( 3 );
    a.append( 3 );
    a.append( 3 );
    a.append( 5 );
    a.append( 3 );
    a.append( 3 );
    a.append( 3 );
    a.append( 5 );
    a.append( 5 );
    a.append( 5 );

    a.shuffle();

    a.deduplicate();

    a.sort();

    BOOST_REQUIRE( a.size() == 4 );
    BOOST_REQUIRE( a[0] == 1 );
    BOOST_REQUIRE( a[1] == 2 );
    BOOST_REQUIRE( a[2] == 3 );
    BOOST_REQUIRE( a[3] == 5 );

    a.reverse();

    BOOST_REQUIRE( a.size() == 4 );
    BOOST_REQUIRE( a[0] == 5 );
    BOOST_REQUIRE( a[1] == 3 );
    BOOST_REQUIRE( a[2] == 2 );
    BOOST_REQUIRE( a[3] == 1 );

    a.save( "array_test_file" );
    a.finalize();
    a.load( "array_test_file" );

    BOOST_REQUIRE( a.size() == 4 );
    BOOST_REQUIRE( a[0] == 5 );
    BOOST_REQUIRE( a[1] == 3 );
    BOOST_REQUIRE( a[2] == 2 );
    BOOST_REQUIRE( a[3] == 1 );

    DeleteFile( "array_test_file" );
}

BOOST_AUTO_TEST_CASE( array_test_05 )
{
    IntArray a;
    a.append( 1 );
    a.append( 2 );
    a.append( 3 );

    BOOST_REQUIRE( a.pop_back() == 3 );

    BOOST_REQUIRE( a.size() == 2 );

    BOOST_REQUIRE( a[0] = 1 );
    BOOST_REQUIRE( a[1] = 2 );
}

BOOST_AUTO_TEST_CASE( array_test_06 )
{
    IndexArray a;
    a.append( 1 );
    a.append( 2 );
    a.append( 3 );

    IndexArray b( 0, kHost    );
    IndexArray c( 0, kDevice  );
    IndexArray d( 0, kDevice  );
    IndexArray e( 0, kUnified );
    IndexArray f( 0, kUnified );
    IndexArray g( 0, kDevice  );
    IndexArray h( 0, kHost    );
    IndexArray i( 0, kHost    );
    IndexArray j( 0, kUnified );
/*
    a.copyTo( b ); // host    -> host
    b.copyTo( c ); // host    -> device
    c.copyTo( d ); // device  -> device
    d.copyTo( e ); // device  -> unified
    e.copyTo( f ); // unified -> unified
    f.copyTo( g ); // unified -> device
    g.copyTo( h ); // device  -> host
    f.copyTo( i ); // unified -> host
    a.copyTo( j ); // host    -> unified

    BOOST_REQUIRE( a.size() == 3 );
    BOOST_REQUIRE( a[0] = 1 );
    BOOST_REQUIRE( a[1] = 2 );
    BOOST_REQUIRE( a[2] = 3 );

    BOOST_REQUIRE( b.size() == 3 );
    BOOST_REQUIRE( b[0] = 1 );
    BOOST_REQUIRE( b[1] = 2 );
    BOOST_REQUIRE( b[2] = 3 );

    BOOST_REQUIRE( e.size() == 3 );
    BOOST_REQUIRE( e[0] = 1 );
    BOOST_REQUIRE( e[1] = 2 );
    BOOST_REQUIRE( e[2] = 3 );

    BOOST_REQUIRE( f.size() == 3 );
    BOOST_REQUIRE( f[0] = 1 );
    BOOST_REQUIRE( f[1] = 2 );
    BOOST_REQUIRE( f[2] = 3 );

    BOOST_REQUIRE( i.size() == 3 );
    BOOST_REQUIRE( i[0] = 1 );
    BOOST_REQUIRE( i[1] = 2 );
    BOOST_REQUIRE( i[2] = 3 );

    BOOST_REQUIRE( j.size() == 3 );
    BOOST_REQUIRE( j[0] = 1 );
    BOOST_REQUIRE( j[1] = 2 );
    BOOST_REQUIRE( j[2] = 3 );

    f.append( 4 );
    f.append( 5 );
    f.append( 6 );

    BOOST_REQUIRE( f.size() == 6 );
    BOOST_REQUIRE( f[0] = 1 );
    BOOST_REQUIRE( f[1] = 2 );
    BOOST_REQUIRE( f[2] = 3 );
    BOOST_REQUIRE( f[3] = 4 );
    BOOST_REQUIRE( f[4] = 5 );
    BOOST_REQUIRE( f[5] = 6 );
*/    
}

BOOST_AUTO_TEST_SUITE_END()

