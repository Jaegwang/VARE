//------------------//
// CudaTestCases.cu //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
//         Wanho Choi @ Dexter Studios                   //
//         Julie Jang @ Dexter Studios                   //
// last update: 2018.04.24                               //
//-------------------------------------------------------//

#include <CudaTestCases.h>

void cudaAddVectors()
{
	TimeWatch watch;

	// a = b + c
	IntArray a, b, c;
	int n = 100000000;
	a.initialize( n, kUnified );
	b.initialize( n, kUnified );
	c.initialize( n, kUnified );

	b.setValueAll( 2 );
	c.setValueAll( 3 );

	int* a_ptr = a.pointer();

	auto kernel = [=] BORA_FUNC_QUAL ( const size_t idx )
	{
		a_ptr[idx] = b[idx] * c[idx];
	};

	watch.start( "Device test" );
	LaunchCudaDevice( kernel, 0, n );
	SyncCuda();
	watch.stop();

	watch.start( "Host test" );
	LaunchCudaHost( kernel, 0, n ); 
	SyncCuda();
	watch.stop();

	//do JuliaTest
}

