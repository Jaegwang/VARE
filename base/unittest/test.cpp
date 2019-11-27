//----------//
// test.cpp //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
//         Wanho Choi @ Dexter Studios                   //
//         Julie Jang @ Dexter Studios                   //
// last update: 2018.04.24                               //
//-------------------------------------------------------//

#define BOOST_TEST_MODULE "Bora Base Unit-Tester"
#include <boost/test/included/unit_test.hpp>

#undef _POSIX_C_SOURCE
#undef _XOPEN_SOURCE

#include <MacroTest.h>
#include <MathUtilsTest.h>
#include <Vector3Test.h>
#include <Vector4Test.h>
#include <Matrix22Test.h>
#include <Matrix33Test.h>
#include <Matrix44Test.h>
#include <ArrayTest.h>
#include <EquationTest.h>
#include <StringArrayTest.h>
#include <JsonTest.h>
#include <CudaTest.h>
