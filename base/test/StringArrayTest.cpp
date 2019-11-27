//---------------//
// ArrayTest.cpp //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.10.26                               //
//-------------------------------------------------------//

#include <Bora.h>

using namespace std;
using namespace Bora;

int main( int argc, char** argv )
{
    StringArray a( 3 );

    a.push_back( "aaa" );
    a.push_back( "bbb" );
    a.push_back( "ccc" );

    a[0] = "111";
    a[1] = "222";
    a[2] = "333";

    a.print();

    StringArray b( a );
    StringArray c;
    c = a;

    c.print();

    c.erase( 3 );
    c.exclude( "111" );
    c.print();

    c.print();

    IndexArray delList;
    delList.append( 0 );
    delList.append( 3 );
    a.remove( delList );

    a.print();

    //cout << (a==c) << " " << (a!=c) << endl;

    a.reverse();
    a.print();

    a.shuffle(123);
    a.print();

    //c.clear();

    a.append( c );
    a.print();

    cout << a.combine() << endl;
    cout << a.combine("_") << endl;

    std::string ccc = a.combine("_");
    a.setByTokenizing( ccc, "_" );
    a.print();

	return 0;
}

