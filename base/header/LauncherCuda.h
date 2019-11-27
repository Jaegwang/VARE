//----------------//
// LauncherCuda.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2019.04.01                               //
//-------------------------------------------------------//

#pragma once

#include <Foundation.h>

template<class KERNEL>
inline void LaunchCudaDevice( KERNEL& kernel, const size_t& P, const size_t& N )
{
    auto r = thrust::counting_iterator<size_t>(P);
 	thrust::for_each( thrust::cuda::par, r, r+N, kernel );
}

template<class KERNEL>
inline void LaunchCudaHost( KERNEL& kernel, const size_t& P, const size_t& N )
{
    auto r = thrust::counting_iterator<size_t>(P);
	thrust::for_each( thrust::cpp::par, r, r+N, kernel );
}

inline void SyncCuda()
{
	cudaDeviceSynchronize();
}

