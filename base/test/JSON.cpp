//----------//
// JSON.cpp //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.03.08                               //
//-------------------------------------------------------//

#include <Bora.h>

using namespace std;
using namespace Bora;

int main( int argc, char** argv )
{
    JSONString js;

    ////////////////////////////////////

    int aaa = 123;
    js.append( "aaa", aaa );

    float bbb = 345;
    js.append( "bbb", bbb );

    Vec3f ccc( 6, 7, 8 );
    js.append( "ccc", ccc );

    string ddd = "abcdefg";
    js.append( "ddd", ddd );

    cout << js.json() << endl;

    ////////////////////////////////////

    aaa = 0;
    js.get( "aaa", aaa );
    cout << "aaa: " << aaa << endl;

    bbb = 0.f;
    js.get( "bbb", bbb );
    cout << "bbb: " << bbb << endl;

    ccc.zeroize();
    js.get( "ccc", ccc );
    cout << "ccc: " << ccc << endl;

    ddd = "";
    js.get( "ddd", ddd );
    cout << "ddd: " << ddd << endl;

    ////////////////////////////////////

    js.save( "aaa" );
    js.clear();
    js.load( "aaa" );

    aaa = 0;
    js.get( "aaa", aaa );
    cout << "aaa: " << aaa << endl;

    bbb = 0.f;
    js.get( "bbb", bbb );
    cout << "bbb: " << bbb << endl;

    ccc.zeroize();
    js.get( "ccc", ccc );
    cout << "ccc: " << ccc << endl;

    ddd = "";
    js.get( "ddd", ddd );
    cout << "ddd: " << ddd << endl;

	return 0;
}

