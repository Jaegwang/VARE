//------------------//
// VarRangeTest.cpp //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.10.19                               //
//-------------------------------------------------------//

#include <Bora.h>

using namespace std;
using namespace Bora;

int main( int argc, char** argv )
{
    // char
    {
        char min = CHAR_MIN;
        char max = CHAR_MAX;

        COUT << "char: " << typeid(char).name() << ENDL;
        COUT << CHAR_MIN << " ~ " << CHAR_MAX << ENDL;
        COUT << (int)min << " ~ " << (int)max << ENDL << ENDL;
    }

    // unsigned char
    {
        unsigned char min = 0;
        unsigned char max = UCHAR_MAX;

        COUT << "unsigned char: " << typeid(unsigned char).name() << ENDL;
        COUT << 0 << " ~ " << UCHAR_MAX << ENDL;
        COUT << (int)min << " ~ " << (int)max << ENDL << ENDL;
    }

    // short
    {
        short min = SHRT_MIN;
        short max = SHRT_MAX;

        COUT << "short: " << typeid(short).name() << ENDL;
        COUT << SHRT_MIN << " ~ " << SHRT_MAX << ENDL;
        COUT << min << " ~ " << max << ENDL << ENDL;
    }

    // unsigned short
    {
        unsigned short min = 0;
        unsigned short max = USHRT_MAX;

        COUT << "unsigned short: " << typeid(unsigned short).name() << ENDL;
        COUT << 0 << " ~ " << USHRT_MAX << ENDL;
        COUT << min << " ~ " << max << ENDL << ENDL;
    }

    // int
    {
        int min = INT_MIN;
        int max = INT_MAX;

        COUT << "int: " << typeid(int).name() << ENDL;
        COUT << INT_MIN << " ~ " << INT_MAX << ENDL;
        COUT << min << " ~ " << max << ENDL << ENDL;
    }

    // unsigned int
    {
        unsigned int min = 0;
        unsigned int max = UINT_MAX;

        COUT << "unsigned int: " << typeid(unsigned int).name() << ENDL;
        COUT << 0 << " ~ " << UINT_MAX << ENDL;
        COUT << min << " ~ " << max << ENDL << ENDL;
    }

    // long int
    {
        long int min = LONG_MIN;
        long int max = LONG_MAX;

        COUT << "long int: " << typeid(long int).name() << ENDL;
        COUT << LONG_MIN << " ~ " << LONG_MAX << ENDL;
        COUT << min << " ~ " << max << ENDL << ENDL;
    }

    // unsigned long int
    {
        unsigned long int min = 0;
        unsigned long int max = ULONG_MAX;

        COUT << "unsigned long int: " << typeid(unsigned long int).name() << ENDL;
        COUT << 0 << " ~ " << ULONG_MAX << ENDL;
        COUT << min << " ~ " << max << ENDL << ENDL;
    }

    // long long int
    {
        long long int min = LLONG_MIN;
        long long int max = LLONG_MAX;

        COUT << "long long int: " << typeid(long long int).name() << ENDL;
        COUT << LLONG_MIN << " ~ " << LLONG_MAX << ENDL;
        COUT << min << " ~ " << max << ENDL << ENDL;
    }

    // unsigned long long int
    {
        unsigned long long int min = 0;
        unsigned long long int max = ULLONG_MAX;

        COUT << "unsigned long long int: " << typeid(unsigned long long int).name() << ENDL;
        COUT << 0 << " ~ " << ULLONG_MAX << ENDL;
        COUT << min << " ~ " << max << ENDL << ENDL;
    }

    // float
    {
        float min = -FLT_MAX;
        float max = FLT_MAX;

        COUT << "float: " << typeid(float).name() << ENDL;
        COUT << FLT_MIN << " ~ " << FLT_MAX << ENDL;
        COUT << min << " ~ " << max << ENDL << ENDL;
    }

    // double
    {
        double min = -DBL_MAX;
        double max = DBL_MAX;

        COUT << "double: " << typeid(double).name() << ENDL;
        COUT << DBL_MIN << " ~ " << DBL_MAX << ENDL;
        COUT << min << " ~ " << max << ENDL << ENDL;
    }

    COUT << "uint32_t: " << sizeof(uint32_t) << ENDL;
    COUT << "uint64_t: " << sizeof(uint64_t) << ENDL;
    COUT << "size_t  : " << sizeof(size_t)  << ENDL;

	return 0;
}

