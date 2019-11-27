//--------------//
// cubicEqn.cpp //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.11.29                               //
//-------------------------------------------------------//

#include <Bora.h>

using namespace std;
using namespace Bora;

int main( int argc, char** argv )
{
    double a=0, b=0, c=0, d=0;

    cout << "Input the coefficients (a,b,c,d): " << endl;
    scanf( "%lf %lf %lf %lf", &a, &b, &c, &d );

    Complex<double> x[3];
    const int nRoots = SolveCubicEqn( a, b, c, d, x );

    cout << "nRoots: " << nRoots << endl;
    cout << x[0] << ": " << (a*Pow3(x[0])+b*Pow2(x[0])+c*x[0]+d).radius() << endl;
    cout << x[1] << ": " << (a*Pow3(x[0])+b*Pow2(x[0])+c*x[0]+d).radius() << endl;
    cout << x[2] << ": " << (a*Pow3(x[0])+b*Pow2(x[0])+c*x[0]+d).radius() << endl;

	return 0;
}

