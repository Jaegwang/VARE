
#include <VARE.h>
#include <EmbeddedPython.h>
#include <iostream>

using namespace VARE;

int main(int argc, char** argv)
{

	FloatArray A, B, C0, C1;
	A.initialize( 10000, kUnified );
	B.initialize( 10000, kUnified );

	for( size_t t=0; t<A.size(); ++t )
	{
		A[t] = Rand( t*423 );
		B[t] = Rand( t*323 );
	}

	C0.initialize( 10000, kUnified );
	C1.initialize( 10000, kUnified );

	float* pA = A.pointer();
	float* pB = B.pointer();
	float* pC0 = C0.pointer();
	float* pC1 = C1.pointer();
	
	auto device_kernel = VARE_DEVICE_KERNEL
	{
		pC0[ix] = pA[ix] + pB[ix];
	};

	LaunchDeviceKernel( device_kernel, 0, C0.size() );
	SyncKernels();

	auto host_kernel = VARE_HOST_KERNEL
	{
		pC1[ix] = pA[ix] + pB[ix];
	};

	LaunchHostKernel(host_kernel, 0, C1.size());
	SyncKernels();

	for (size_t t = 0; t<A.size(); ++t)
	{
		if( C0[t] != C1[t] )
		{
			std::cout<<"Error"<<std::endl;
			break;
		}
	}


	while (true);

	return 0;
}

