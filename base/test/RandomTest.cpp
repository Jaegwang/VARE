//----------------//
// RandomTest.cpp //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.10.19                               //
//-------------------------------------------------------//

#include <Bora.h>

using namespace std;
using namespace Bora;

int main( int argc, char** argv )
{
    const int seed = 0;

    for( int i=0; i<100; ++i )
    {
        //COUT << i << ": " << Rand( seed + i ) << ENDL;
        COUT << i << ": " << RandInt( seed+i, 0, 100 ) << ENDL;
    }

	return 0;
}

