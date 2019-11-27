//----------------//
// Matrix22Test.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.11.27                               //
//-------------------------------------------------------//

#include <Bora.h>
using namespace std;
using namespace Bora;

BOOST_AUTO_TEST_SUITE( Matrix22Suite )

BOOST_AUTO_TEST_CASE( matrix22_structure )
{
    Mat22f a;
    BOOST_REQUIRE( sizeof(a) == (4*sizeof(float)) );
    BOOST_REQUIRE( a.isIdentity() );

    Mat22f m( 1,2, 3,4 );

    BOOST_REQUIRE( m._00 == 1 );
    BOOST_REQUIRE( m._01 == 2 );
    BOOST_REQUIRE( m._10 == 3 );
    BOOST_REQUIRE( m._11 == 4 );

    BOOST_REQUIRE( m.v[0] == 1 );
    BOOST_REQUIRE( m.v[1] == 3 );
    BOOST_REQUIRE( m.v[2] == 2 );
    BOOST_REQUIRE( m.v[3] == 4 );

    BOOST_REQUIRE( m._00 == m.column[0][0] );
    BOOST_REQUIRE( m._01 == m.column[1][0] );
    BOOST_REQUIRE( m._10 == m.column[0][1] );
    BOOST_REQUIRE( m._11 == m.column[1][1] );

    BOOST_REQUIRE( m.row(0) == Vec2f(1,2) );
    BOOST_REQUIRE( m.row(1) == Vec2f(3,4) );

    BOOST_REQUIRE( m.column[0] == Vec2f(1,3) );
    BOOST_REQUIRE( m.column[1] == Vec2f(2,4) );

    BOOST_REQUIRE( m[0] == Vec2f(1,3) );
    BOOST_REQUIRE( m[1] == Vec2f(2,4) );
}

BOOST_AUTO_TEST_CASE( matrix22_inverse )
{
    const float values[4] = { 1,2, 2,3 };

    Mat22f A( values );
    Mat22f inv_A = A.inversed();

    Mat22f B = A * inv_A;
    BOOST_REQUIRE( B.isIdentity(0.001f) );
    BOOST_REQUIRE( B.isSymmetric(0.001f) );

    Mat22f C = inv_A * A;
    BOOST_REQUIRE( C.isIdentity(0.001f) );
    BOOST_REQUIRE( C.isSymmetric(0.001f) );
}

BOOST_AUTO_TEST_CASE( matrix22_eigen )
{
    Mat22f M( 1, 2, 2, 1 );

    float eigenvalues[2];
    Vec2f eigenvectors[2];

    M.eigen( eigenvalues, eigenvectors );

    Mat22f L( eigenvalues[0], eigenvalues[1] );
    Mat22f P( eigenvectors[0], eigenvectors[1] );

    BOOST_REQUIRE( M.isSame( P*L*P.transposed(), 0.001f ) );
}

BOOST_AUTO_TEST_SUITE_END()

