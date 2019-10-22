#pragma once

#define VARE_NAMESPACE_BEGIN namespace VARE{
#define VARE_NAMESPACE_END   }

#define VARE_UNIFIED __host__ __device__
#define VARE_DEVICE  __device__
#define VARE_HOST    __host__
#define VARE_INDEX   size_t

#define COUT std::cout
#define ENDL std::endl

// C++ Standards
#include <iostream>
#include <vector>
#include <stack>
#include <map>
#include <thread>
#include <algorithm>
#include <fstream>
#include <ctime>
#include <string>
#include <sstream>
#include <fstream>

// CUDA
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/inner_product.h>

// glm
#include <glm/glm.hpp>
#include <glm/detail/type_mat3x3.hpp>
#include <glm/detail/type_mat4x4.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/noise.hpp>
#include <glm/gtc/random.hpp>

// OpenVDB
#include <openvdb/openvdb.h>
#include <openvdb/tools/ChangeBackground.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/tools/GridOperators.h>


VARE_NAMESPACE_BEGIN

// PRE-DEFINES
enum MemorySpace { kHost, kDevice, kUnified };
enum VoxelSpace  { kEmpty, kSolid, kWall, kMark, kFluid };

VARE_NAMESPACE_END

#define _DX 1.f
#define _DY 1.f
#define _DZ 1.f
