//----------------//
// Matrix44Test.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.11.27                               //
//-------------------------------------------------------//

#include <Bora.h>
using namespace std;
using namespace Bora;

BOOST_AUTO_TEST_SUITE( Matrix44Suite )

BOOST_AUTO_TEST_CASE( matrix44_structure )
{
    Mat44f a;
    BOOST_REQUIRE( sizeof(a) == (16*sizeof(float)) );
    BOOST_REQUIRE( a.isIdentity() );

    Mat44f m( 1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16 );

    BOOST_REQUIRE( m._00 ==  1 );
    BOOST_REQUIRE( m._01 ==  2 );
    BOOST_REQUIRE( m._02 ==  3 );
    BOOST_REQUIRE( m._03 ==  4 );
    BOOST_REQUIRE( m._10 ==  5 );
    BOOST_REQUIRE( m._11 ==  6 );
    BOOST_REQUIRE( m._12 ==  7 );
    BOOST_REQUIRE( m._13 ==  8 );
    BOOST_REQUIRE( m._20 ==  9 );
    BOOST_REQUIRE( m._21 == 10 );
    BOOST_REQUIRE( m._22 == 11 );
    BOOST_REQUIRE( m._23 == 12 );
    BOOST_REQUIRE( m._30 == 13 );
    BOOST_REQUIRE( m._31 == 14 );
    BOOST_REQUIRE( m._32 == 15 );
    BOOST_REQUIRE( m._33 == 16 );

    BOOST_REQUIRE( m.v[ 0] ==  1 );
    BOOST_REQUIRE( m.v[ 1] ==  5 );
    BOOST_REQUIRE( m.v[ 2] ==  9 );
    BOOST_REQUIRE( m.v[ 3] == 13 );
    BOOST_REQUIRE( m.v[ 4] ==  2 );
    BOOST_REQUIRE( m.v[ 5] ==  6 );
    BOOST_REQUIRE( m.v[ 6] == 10 );
    BOOST_REQUIRE( m.v[ 7] == 14 );
    BOOST_REQUIRE( m.v[ 8] ==  3 );
    BOOST_REQUIRE( m.v[ 9] ==  7 );
    BOOST_REQUIRE( m.v[10] == 11 );
    BOOST_REQUIRE( m.v[11] == 15 );
    BOOST_REQUIRE( m.v[12] ==  4 );
    BOOST_REQUIRE( m.v[13] ==  8 );
    BOOST_REQUIRE( m.v[14] == 12 );
    BOOST_REQUIRE( m.v[15] == 16 );

    BOOST_REQUIRE( m._00 == m.column[0][0] );
    BOOST_REQUIRE( m._01 == m.column[1][0] );
    BOOST_REQUIRE( m._02 == m.column[2][0] );
    BOOST_REQUIRE( m._03 == m.column[3][0] );
    BOOST_REQUIRE( m._10 == m.column[0][1] );
    BOOST_REQUIRE( m._11 == m.column[1][1] );
    BOOST_REQUIRE( m._12 == m.column[2][1] );
    BOOST_REQUIRE( m._13 == m.column[3][1] );
    BOOST_REQUIRE( m._20 == m.column[0][2] );
    BOOST_REQUIRE( m._21 == m.column[1][2] );
    BOOST_REQUIRE( m._22 == m.column[2][2] );
    BOOST_REQUIRE( m._23 == m.column[3][2] );
    BOOST_REQUIRE( m._30 == m.column[0][3] );
    BOOST_REQUIRE( m._31 == m.column[1][3] );
    BOOST_REQUIRE( m._32 == m.column[2][3] );
    BOOST_REQUIRE( m._33 == m.column[3][3] );

    BOOST_REQUIRE( m.row(0) == Vec4f( 1, 2, 3, 4) );
    BOOST_REQUIRE( m.row(1) == Vec4f( 5, 6, 7, 8) );
    BOOST_REQUIRE( m.row(2) == Vec4f( 9,10,11,12) );
    BOOST_REQUIRE( m.row(3) == Vec4f(13,14,15,16) );

    BOOST_REQUIRE( m.column[0] == Vec4f( 1, 5, 9,13) );
    BOOST_REQUIRE( m.column[1] == Vec4f( 2, 6,10,14) );
    BOOST_REQUIRE( m.column[2] == Vec4f( 3, 7,11,15) );
    BOOST_REQUIRE( m.column[3] == Vec4f( 4, 8,12,16) );

    BOOST_REQUIRE( m[0] == Vec4f( 1, 5, 9,13) );
    BOOST_REQUIRE( m[1] == Vec4f( 2, 6,10,14) );
    BOOST_REQUIRE( m[2] == Vec4f( 3, 7,11,15) );
    BOOST_REQUIRE( m[3] == Vec4f( 4, 8,12,16) );
}

BOOST_AUTO_TEST_CASE( matrix44_inverse )
{
    const float values[16] = { 1,2,3,4, 3,5,4,7, 5,8,6,10, 10,11,12,13 };

    Mat44f A( values );
    Mat44f inv_A = A.inversed();

    Mat44f B = A * inv_A;
    BOOST_REQUIRE( B.isIdentity(0.001f) );
    BOOST_REQUIRE( B.isSymmetric(0.001f) );

    Mat44f C = inv_A * A;
    BOOST_REQUIRE( C.isIdentity(0.001f) );
    BOOST_REQUIRE( C.isSymmetric(0.001f) );
}

BOOST_AUTO_TEST_SUITE_END()

