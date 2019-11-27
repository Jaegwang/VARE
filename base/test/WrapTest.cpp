//--------------//
// WrapTest.cpp //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.01.03                               //
//-------------------------------------------------------//

#include <Bora.h>

using namespace std;
using namespace Bora;

int main( int argc, char** argv )
{
    COUT << Wrap( 1, 3 ) << ENDL;
    COUT << Wrap( 5, 3 ) << ENDL;
    COUT << Wrap( 6, 3 ) << ENDL;
    COUT << Wrap( -1, 3 ) << ENDL;
    COUT << ENDL;

    COUT << Wrap( 0.1f, 2.0f ) << ENDL;
    COUT << Wrap( 2.1f, 2.0f ) << ENDL;
    COUT << Wrap( 5.1f, 2.0f ) << ENDL;
    COUT << Wrap( -0.1f, 2.0f ) << ENDL;
    COUT << ENDL;

    COUT << Wrap( 0.1, 2.0 ) << ENDL;
    COUT << Wrap( 2.1, 2.0 ) << ENDL;
    COUT << Wrap( 5.1, 2.0 ) << ENDL;
    COUT << Wrap( -0.1, 2.0 ) << ENDL;
    COUT << ENDL;

	return 0;
}

