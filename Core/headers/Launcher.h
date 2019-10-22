/**
_______________
 VARE Launcher |
	@file    Launcher.h
	@author  Jaegwang Lim
	@date    2014-04-23
	@version 2014-04-23
*/

#pragma once
#include <Commons.h>

#define VARE_HOST_KERNEL    [&] __host__ ( const size_t ix )
#define VARE_DEVICE_KERNEL  [=] __device__ ( const size_t ix )
#define VARE_UNIFIED_KERNEL [=] __host__ __device__ ( const size_t ix )
//[=](index<1> i) [[hc]]

static std::vector< std::thread > _launcherThreads;

template<class TT>
void LaunchDeviceKernel( TT& kernel, VARE_INDEX first, VARE_INDEX numElements )
{
	_launcherThreads.push_back(std::thread([&]() {
	
		thrust::counting_iterator<VARE_INDEX> F(first);
		thrust::for_each(thrust::device, F, F + numElements, kernel);
	
	}));
}

template<class TT>
void LaunchHostKernel( TT& kernel, VARE_INDEX first, VARE_INDEX numElements )
{
	size_t numThreads = std::min((size_t)std::thread::hardware_concurrency(), numElements / (size_t)100 + 1);

	for (size_t q = 0; q < numThreads; ++q)
	{
		_launcherThreads.push_back(std::thread([&]() {

			size_t F = q * numElements / numThreads;
			size_t L = F + numElements / numThreads;

			if (q == 0) L += numElements % numThreads;

			for (size_t n = F; n < L; ++n) kernel(first + n);

		}));
	}

}

inline void SyncKernels()
{
	for (size_t q = 0; q < _launcherThreads.size(); ++q) _launcherThreads[q].join();
	_launcherThreads.clear();
}

