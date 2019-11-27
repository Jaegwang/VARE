//-------------------//
// NumericLimits.cpp //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.02.22                               //
//-------------------------------------------------------//

#include <Bora.h>

using namespace std;
using namespace Bora;

int main( int argc, char** argv )
{
    cout <<       "1" << " " << numeric_limits<bool>::max()           << endl;
    cout << UCHAR_MAX << " " << numeric_limits<unsigned char>::max()  << endl;
    cout <<  SHRT_MAX << " " << numeric_limits<short>::max()          << endl;
    cout << USHRT_MAX << " " << numeric_limits<unsigned short>::max() << endl;
    cout <<   INT_MAX << " " << numeric_limits<int>::max()            << endl;
    cout <<  UINT_MAX << " " << numeric_limits<unsigned int>::max()   << endl;
    cout <<  LONG_MAX << " " << numeric_limits<long>::max()           << endl;
    cout << ULONG_MAX << " " << numeric_limits<unsigned long>::max()  << endl;
    cout <<   FLT_MAX << " " << numeric_limits<float>::max()          << endl;
    cout <<   DBL_MAX << " " << numeric_limits<double>::max()         << endl;
    cout <<  LDBL_MAX << " " << numeric_limits<long double>::max()    << endl;

	return 0;
}

