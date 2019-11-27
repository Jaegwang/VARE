//--------------//
// Foundation.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2019.04.11                               //
//-------------------------------------------------------//

#pragma once

#define BORA_FUNC_QUAL __device__ __host__
#define BORA_UNIFIED   __device__ __host__
#define BORA_DEVICE    __device__
#define BORA_HOST      __host__

#define BORA_NAMESPACE_BEGIN namespace Bora {
#define BORA_NAMESPACE_END   }

#define BORA_HOST_KERNEL   [&] __host__ ( const size_t ix )
#define BORA_DEVICE_KERNEL [=] __device__ ( const size_t ix )

// C++ Standards
#include <map>
#include <set>
#include <list>
#include <stack>
#include <queue>
#include <cmath>
#include <bitset>
#include <vector>
#include <thread> 
#include <vector>
#include <string>
#include <atomic>
#include <cstdarg>
#include <fstream>
#include <iomanip>
#include <errno.h>
#include <netdb.h>
#include <climits>
#include <limits.h>
#include <stdlib.h>
#include <iostream>
#include <ostream>
#include <iterator>
#include <dirent.h>
#include <stdint.h>
#include <ifaddrs.h>
#include <algorithm>
#include <chrono>
#include <unordered_set>
#include <unordered_map>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/utsname.h>

// TBB
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/mutex.h>

// FFTW
#include <fftw3.h>

// OpenGL
#include <GL/glew.h>
#include <GL/glut.h>

// OpenEXR
#include <OpenEXR/ImfRgbaFile.h>

// JPEG
#include <jpeglib.h>

// TIFF
#include <tiffio.h>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

// Thrust
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>
#include <thrust/extrema.h>

// glm
#include <glm/glm.hpp>
#include <glm/detail/type_mat3x3.hpp>
#include <glm/detail/type_mat4x4.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/noise.hpp>
#include <glm/gtc/random.hpp>

// OpenVDB 4.0
#include <openvdb/openvdb.h>
#include <openvdb/tools/ChangeBackground.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/tools/GridOperators.h>

