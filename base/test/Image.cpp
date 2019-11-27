//-----------//
// Image.cpp //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.03.23                               //
//-------------------------------------------------------//

#include <Bora.h>

using namespace std;
using namespace Bora;

int main( int argc, char** argv )
{
    Image img;

    //img.load( "/home/wanho.choi/work/Zelos/data/img/uffizi.exr" );
    //img.load( "/home/wanho.choi/work/Zelos/data/img/RGB.jpg" );
    img.load( "/home/wanho.choi/work/Zelos/data/img/Stonewall_2k.hdr" );

    cout << img.width() << " x " << img.height() << endl;
    cout << img.compression() << endl;

    for( int j=100; j<300; ++j )
    {
        for( int i=100; i<500; ++i )
        {
            Pixel& p = img( i, j );

            p.r = 1.0;
            p.g = 0.0;
            p.b = 0.0;
            p.a = 1.0;
        }
    }

    //img.save( "result.exr" );
    //img.save( "result.jpg" );
    img.save( "result.hdr" );

	return 0;
}

