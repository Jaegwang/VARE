//---------------//
// Definitions.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2018.04.12                               //
//-------------------------------------------------------//

#ifndef _BoraDefinitions_h_
#define _BoraDefinitions_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

#define COUT std::cout
#define ENDL std::endl

#define EPSILON  1e-30f
#define INFINITE 1e+30f

#define MAX_SIZE_T 0xffffffffffffffff
#define NULL_MAX   0xffffffffffffffff

#define _DX 1.f
#define _DY 1.f
#define _DZ 1.f
#define _INV_DX 1.f
#define _INV_DY 1.f
#define _INV_DZ 1.f

#define PI 3.1415926535897931159979634685441851615906f  // pi
#define _PI 0.3183098861837906912164442019275156781077f // 1/pi
#define PI2 6.2831853071795862319959269370883703231812f // 2*pi
#define PI3 9.4247779607693793479938904056325554847717f // 3*pi
#define PI_2 1.5707963267948965579989817342720925807953f ///< pi/2
#define PI_3 1.0471975511965976313177861811709590256214f ///< pi/3
#define PI_4 0.7853981633974482789994908671360462903976f ///< pi/4
#define PI_6 0.5235987755982988156588930905854795128107f ///< pi/6

#define SQRT2 1.4142135623730951454746218587388284504414f // sqrt(2)
#define SQRT3 1.7320508075688771931766041234368458390236f // sqrt(3)

#define STRINGER(x) #x

enum MemorySpace
{
    kHost,
    kDevice,
    kUnified
};

enum Axis
{
    kXaxis,
    kYaxis,
    kZaxis
};

// Maya's default rotation order is kXYZ.
// : q^T = p^T * (Rx * Ry * Rz);
// In Bora, the corresponding order is kZYX,
// because Bora uses row-major ordering matrix system while Maya uses colume-major ordering matrix system.
// (One is the transpose matrix of the other.)
// Therefore, after applying a rotation matrix,
// Bora: counter-clockwise rotation (i.e. right handed coordinate system)
// Maya: clockwise rotation (i.e. left handed coordinate system)

// For the same effect as Maya,
// Use kZYX, and then transpose the matrix.
enum RotationOrder
{
    kXYZ = 0, // q = (XYZ)p = X(Y(Zp)): Z->Y->X
    kYZX = 1, // q = (YZX)p = Y(Z(Xp)): X->Z->Y
    kZXY = 2, // q = (ZXY)p = Z(X(Yp)): Y->X->Z
    kXZY = 3, // q = (XZY)p = X(Z(Yp)): Y->Z->X
    kYXZ = 4, // q = (YXZ)p = Y(X(Zp)): Z->X->Y
    kZYX = 5  // q = (ZYX)p = Z(Y(Xp)): X->y->Z
};

enum CellType
{
    kFluid     = 0, // fluid (=interior) cell
    kDirichlet = 1, // Dirichlet cell
    kNeumann   = 2, // Neumann cell
};

enum ProjectionScheme
{
    kJacobi = 0, // Jacibo iteration
};

enum DataType
{
    kNone      =  0, // none
    kBool      =  1, // boolean (1 bit) = bool
    kChar      =  2, // signed char (1 byte) = char
    kUChar     =  3, // unsigned char (1 byte) = unsigned char
    kInt16     =  4, // signed integer (2 bytes) = short
    kInt32     =  5, // signed integer (4 bytes) = int = int32_t
    kInt64     =  6, // signed integer (8 bytes) = long int = int64_t
    kUInt16    =  7, // unsigned integer (2 bytes) = unsigned short
    kUInt32    =  8, // unsigned integer (4 bytes) = unsigned int = uint32_t
    kUInt64    =  9, // unsigned integer (8 bytes) = unsigned long int = uint64_t
    kFloat16   = 10, // half-precision real floating-type (2 bytes) = half
    kFloat32   = 11, // single-precision real floating-type (4 bytes) = float = float32_t
    kFloat64   = 12, // double-precision real floating-type (8 bytes) = double = float64_t
    kFloat128  = 13, // extended-precision real floating-type (16 bytes) = long double
    kIdx2      = 14, // 2D index vector type = Vector2<size_t>
    kVec2i     = 15, // 2D int vector type = Vector2<int>
    kVec2f     = 16, // 2D float vector type = Vector2<float>
    kVec2d     = 17, // 2D double vector type = Vector2<double>
    kIdx3      = 18, // 3D index vector type = Vector3<size_t>
    kVec3i     = 19, // 3D int vector type = Vector3<int>
    kVec3f     = 20, // 3D float vector type = Vector3<float>
    kVec3d     = 21, // 3D double vector type = Vector3<double>
    kIdx4      = 22, // 4D index vector type = Vector4<size_t>
    kVec4i     = 23, // 4D int vector type = Vector4<int>
    kVec4f     = 24, // 4D float vector type = Vector4<float>
    kVec4d     = 25, // 4D double vector type = Vector4<double>
    kQuatf     = 26, // float quaternion = Quaternion<float>
    kQuatd     = 27, // double quaternion = Quaternion<double>
    kMat22f    = 28, // 2 by 2 float matrix = Matrix22<float>
    kMat22d    = 29, // 2 by 2 float matrix = Matrix22<double>
    kMat33f    = 30, // 3 by 3 float matrix = Matrix33<float>
    kMat33d    = 31, // 3 by 3 float matrix = Matrix33<double>
    kMat44f    = 32, // 4 by 4 float matrix = Matrix44<float>
    kMat44d    = 33, // 4 by 4 float matrix = Matrix44<double>
};

enum AdvectionScheme
{
    kLinear 	= 0, // semi-Langrangian
    kRK2		= 1, // Runge-Kutta first  order
    kRK3		= 2, // Runge-Kutta second order
    kRK4		= 3, // Runge-Kutta third  order
    kMacCormack	= 4, // MacCormack
    kBFECC		= 5, // BFECC
};

static std::string DataName( const DataType dataType )
{
    switch( dataType )
    {
        case kNone:     { return "none";     }
        case kBool:     { return "bool";     }
        case kChar:     { return "char";     }
        case kUChar:    { return "uchar";    }
        case kInt16:    { return "int16";    }
        case kInt32:    { return "int32";    }
        case kInt64:    { return "int64";    }
        case kUInt16:   { return "uint16";   }
        case kUInt32:   { return "uint32";   }
        case kUInt64:   { return "uint64";   }
        case kFloat16:  { return "float16";  }
        case kFloat32:  { return "float32";  }
        case kFloat64:  { return "float64";  }
        case kFloat128: { return "float128"; }
        case kIdx2:     { return "idx2";     }
        case kVec2i:    { return "vec2i";    }
        case kVec2f:    { return "vec2f";    }
        case kVec2d:    { return "vec2d";    }
        case kIdx3:     { return "idx3";     }
        case kVec3i:    { return "vec3i";    }
        case kVec3f:    { return "vec3f";    }
        case kVec3d:    { return "vec3d";    }
        case kIdx4:     { return "idx4";     }
        case kVec4i:    { return "vec4i";    }
        case kVec4f:    { return "vec4f";    }
        case kVec4d:    { return "vec4d";    }
        case kQuatf:    { return "quatf";    }
        case kQuatd:    { return "quatd";    }
        case kMat22f:   { return "mat22f";   }
        case kMat22d:   { return "mat22d";   }
        case kMat33f:   { return "mat33f";   }
        case kMat33d:   { return "mat33d";   }
        case kMat44f:   { return "mat44f";   }
        case kMat44d:   { return "mat44d";   }
        default:        { return "unknown";  }
    }

    return "";
}

static int DataBytes( const DataType dataType )
{
    switch( dataType )
    {
        case kNone:     { return   0; }
        case kBool:     { return   1; } // sizeof(bool) = 1
        case kChar:     { return   1; }
        case kUChar:    { return   1; }
        case kInt16:    { return   2; }
        case kInt32:    { return   4; }
        case kInt64:    { return   8; }
        case kUInt16:   { return   2; }
        case kUInt32:   { return   4; }
        case kUInt64:   { return   8; }
        case kFloat16:  { return   2; }
        case kFloat32:  { return   4; }
        case kFloat64:  { return   8; }
        case kFloat128: { return  16; }
        case kIdx2:     { return  16; }
        case kVec2i:    { return   8; }
        case kVec2f:    { return   8; }
        case kVec2d:    { return  16; }
        case kIdx3:     { return  24; }
        case kVec3i:    { return  12; }
        case kVec3f:    { return  12; }
        case kVec3d:    { return  24; }
        case kIdx4:     { return  32; }
        case kVec4i:    { return  16; }
        case kVec4f:    { return  16; }
        case kVec4d:    { return  32; }
        case kQuatf:    { return  16; }
        case kQuatd:    { return  32; }
        case kMat22f:   { return  16; }
        case kMat22d:   { return  32; }
        case kMat33f:   { return  36; }
        case kMat33d:   { return  72; }
        case kMat44f:   { return  64; }
        case kMat44d:   { return 128; }
        default:        { return   0; }
    }
    return 0;
}

static int DataExtent( const DataType dataType )
{
    switch( dataType )
    {
        case kNone:     { return  0; }
        case kBool:     { return  1; }
        case kChar:     { return  1; }
        case kUChar:    { return  1; }
        case kInt16:    { return  1; }
        case kInt32:    { return  1; }
        case kInt64:    { return  1; }
        case kUInt16:   { return  1; }
        case kUInt32:   { return  1; }
        case kUInt64:   { return  1; }
        case kFloat16:  { return  1; }
        case kFloat32:  { return  1; }
        case kFloat64:  { return  1; }
        case kFloat128: { return  1; }
        case kIdx2:     { return  2; }
        case kVec2i:    { return  2; }
        case kVec2f:    { return  2; }
        case kVec2d:    { return  2; }
        case kIdx3:     { return  3; }
        case kVec3i:    { return  3; }
        case kVec3f:    { return  3; }
        case kVec3d:    { return  3; }
        case kIdx4:     { return  4; }
        case kVec4i:    { return  4; }
        case kVec4f:    { return  4; }
        case kVec4d:    { return  4; }
        case kQuatf:    { return  4; }
        case kQuatd:    { return  4; }
        case kMat22f:   { return  4; }
        case kMat22d:   { return  4; }
        case kMat33f:   { return  9; }
        case kMat33d:   { return  9; }
        case kMat44f:   { return 16; }
        case kMat44d:   { return 16; }
        default:        { return  0; }
    }
    return 0;
}

BORA_NAMESPACE_END

#endif

