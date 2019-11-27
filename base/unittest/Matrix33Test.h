//----------------//
// Matrix33Test.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2017.11.28                               //
//-------------------------------------------------------//

#include <Bora.h>
using namespace std;
using namespace Bora;

BOOST_AUTO_TEST_SUITE( Matrix33Suite )

BOOST_AUTO_TEST_CASE( matrix33_structure )
{
    Mat33f a;
    BOOST_REQUIRE( sizeof(a) == (9*sizeof(float)) );
    BOOST_REQUIRE( a.isIdentity() );

    Mat33f m( 1,2,3, 4,5,6, 7,8,9 );

    BOOST_REQUIRE( m._00 == 1 );
    BOOST_REQUIRE( m._01 == 2 );
    BOOST_REQUIRE( m._02 == 3 );
    BOOST_REQUIRE( m._10 == 4 );
    BOOST_REQUIRE( m._11 == 5 );
    BOOST_REQUIRE( m._12 == 6 );
    BOOST_REQUIRE( m._20 == 7 );
    BOOST_REQUIRE( m._21 == 8 );
    BOOST_REQUIRE( m._22 == 9 );

    BOOST_REQUIRE( m.v[0] == 1 );
    BOOST_REQUIRE( m.v[1] == 4 );
    BOOST_REQUIRE( m.v[2] == 7 );
    BOOST_REQUIRE( m.v[3] == 2 );
    BOOST_REQUIRE( m.v[4] == 5 );
    BOOST_REQUIRE( m.v[5] == 8 );
    BOOST_REQUIRE( m.v[6] == 3 );
    BOOST_REQUIRE( m.v[7] == 6 );
    BOOST_REQUIRE( m.v[8] == 9 );

    BOOST_REQUIRE( m._00 == m.column[0][0] );
    BOOST_REQUIRE( m._01 == m.column[1][0] );
    BOOST_REQUIRE( m._02 == m.column[2][0] );
    BOOST_REQUIRE( m._10 == m.column[0][1] );
    BOOST_REQUIRE( m._11 == m.column[1][1] );
    BOOST_REQUIRE( m._12 == m.column[2][1] );
    BOOST_REQUIRE( m._20 == m.column[0][2] );
    BOOST_REQUIRE( m._21 == m.column[1][2] );
    BOOST_REQUIRE( m._22 == m.column[2][2] );

    BOOST_REQUIRE( m.row(0) == Vec3f(1,2,3) );
    BOOST_REQUIRE( m.row(1) == Vec3f(4,5,6) );
    BOOST_REQUIRE( m.row(2) == Vec3f(7,8,9) );

    BOOST_REQUIRE( m.column[0] == Vec3f(1,4,7) );
    BOOST_REQUIRE( m.column[1] == Vec3f(2,5,8) );
    BOOST_REQUIRE( m.column[2] == Vec3f(3,6,9) );

    BOOST_REQUIRE( m[0] == Vec3f(1,4,7) );
    BOOST_REQUIRE( m[1] == Vec3f(2,5,8) );
    BOOST_REQUIRE( m[2] == Vec3f(3,6,9) );
}

BOOST_AUTO_TEST_CASE( matrix33_inverse )
{
    const float values[9] = { 1,2,3, 0,1,5, 5,6,0 };

    Mat33f A( values );
    Mat33f inv_A = A.inversed();

    Mat33f B = A * inv_A;
    BOOST_REQUIRE( B.isIdentity(0.001f) );
    BOOST_REQUIRE( B.isSymmetric(0.001f) );

    Mat33f C = inv_A * A;
    BOOST_REQUIRE( C.isIdentity(0.001f) );
    BOOST_REQUIRE( C.isSymmetric(0.001f) );
}

BOOST_AUTO_TEST_CASE( matrix33_transformaion )
{
    float rx = DegToRad( -30 );
    float ry = DegToRad(  70 );
    float rz = DegToRad( 110 );

    float rrx=0.f, rry=0.f, rrz=0.f;

    Mat33f Rx, Ry, Rz;

    Rx.setRotationAboutX( rx );
    Ry.setRotationAboutY( ry );
    Rz.setRotationAboutZ( rz );

    Mat33f Rxyz = Rx * Ry * Rz;
    Mat33f Ryzx = Ry * Rz * Rx;
    Mat33f Rzxy = Rz * Rx * Ry;
    Mat33f Rxzy = Rx * Rz * Ry;
    Mat33f Ryxz = Ry * Rx * Rz;
    Mat33f Rzyx = Rz * Ry * Rx;

    Mat33f R;

    R.setRotation( rx,ry,rz, RotationOrder::kXYZ );
    BOOST_REQUIRE( R.isSame( Rxyz, 0.001f ) );

    R.setRotation( rx,ry,rz, RotationOrder::kYZX );
    BOOST_REQUIRE( R.isSame( Ryzx, 0.001f ) );

    R.setRotation( rx,ry,rz, RotationOrder::kZXY );
    BOOST_REQUIRE( R.isSame( Rzxy, 0.001f ) );

    R.setRotation( rx,ry,rz, RotationOrder::kXZY );
    BOOST_REQUIRE( R.isSame( Rxzy, 0.001f ) );

    R.setRotation( rx,ry,rz, RotationOrder::kYXZ );
    BOOST_REQUIRE( R.isSame( Ryxz, 0.001f ) );

    R.setRotation( rx,ry,rz, RotationOrder::kZYX );
    BOOST_REQUIRE( R.isSame( Rzyx, 0.001f ) );

    R.getRotation( rrx, rry, rrz );
    BOOST_REQUIRE( AlmostSame( rrx, rx, 0.001f ) );
    BOOST_REQUIRE( AlmostSame( rry, ry, 0.001f ) );
    BOOST_REQUIRE( AlmostSame( rrz, rz, 0.001f ) );

    R.addScale( 11.f, 22.f, 33.f );

    float sx=0.f, sy=0.f, sz=0.f;
    R.getScale( sx, sy, sz );

    BOOST_REQUIRE( AlmostSame( sx, 11.f, 0.001f ) );
    BOOST_REQUIRE( AlmostSame( sy, 22.f, 0.001f ) );
    BOOST_REQUIRE( AlmostSame( sz, 33.f, 0.001f ) );

    R.eliminateScale();

    R.getRotation( rrx, rry, rrz );
    BOOST_REQUIRE( AlmostSame( rrx, rx, 0.001f ) );
    BOOST_REQUIRE( AlmostSame( rry, ry, 0.001f ) );
    BOOST_REQUIRE( AlmostSame( rrz, rz, 0.001f ) );
}

BOOST_AUTO_TEST_CASE( matrix33_cofactor )
{
    Mat33f A( 1,2,3, 0,4,5, 1,0,6 );

    BOOST_REQUIRE( AlmostSame( A.cofactor(0,0),  24.f, 0.001f ) );
    BOOST_REQUIRE( AlmostSame( A.cofactor(0,1),   5.f, 0.001f ) );
    BOOST_REQUIRE( AlmostSame( A.cofactor(0,2),  -4.f, 0.001f ) );

    BOOST_REQUIRE( AlmostSame( A.cofactor(1,0), -12.f, 0.001f ) );
    BOOST_REQUIRE( AlmostSame( A.cofactor(1,1),   3.f, 0.001f ) );
    BOOST_REQUIRE( AlmostSame( A.cofactor(1,2),   2.f, 0.001f ) );

    BOOST_REQUIRE( AlmostSame( A.cofactor(2,0),  -2.f, 0.001f ) );
    BOOST_REQUIRE( AlmostSame( A.cofactor(2,1),  -5.f, 0.001f ) );
    BOOST_REQUIRE( AlmostSame( A.cofactor(2,2),   4.f, 0.001f ) );
}

BOOST_AUTO_TEST_SUITE_END()

