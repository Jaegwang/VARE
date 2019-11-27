//------------//
// CudaTest.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
//         Wanho Choi @ Dexter Studios                   //
//         Julie Jang @ Dexter Studios                   //
// last update: 2018.04.24                               //
//-------------------------------------------------------//

#include <Bora.h>
using namespace std;
using namespace Bora;

#include <CudaTestCases.h>

BOOST_AUTO_TEST_SUITE( CudaTestSuite )

BOOST_AUTO_TEST_CASE( cuda_addVectors )
{
    cudaAddVectors();
}

BOOST_AUTO_TEST_SUITE_END()

