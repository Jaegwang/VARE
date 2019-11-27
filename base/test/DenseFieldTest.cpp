//----------//
// Test.cpp //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.01.16                               //
//-------------------------------------------------------//

#include <Bora.h>

using namespace std;
using namespace Bora;

int main( int argc, char** argv )
{
    Grid grid( 10, 10, 10, AABB3f( Vec3f(-1), Vec3f(1) ) );

    ScalarDenseField s;

    s.initialize( grid );
    s.setRandomValues( 1.f, 10.f );

    COUT << s.min() << " ~ " << s.max() << ENDL;

	return 0;
}

