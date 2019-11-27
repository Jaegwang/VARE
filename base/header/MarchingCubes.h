//-----------------//
// MarchingCubes.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.03.19                               //
//-------------------------------------------------------//

#ifndef _BoraMarchingCubes_h_
#define _BoraMarchingCubes_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

class MarchingCubes
{
    private:

        static const int _edgeTable[256];
        static const int _triTable[256][16];

    public:

        static int*  _pEdgeTable;
        static int** _pTriTable;

        float isoValue=0.f;

    public:
    
        BORA_FUNC_QUAL static Vec3f
        vertexInterpolate( const Vec3f& p0, const Vec3f& p1, const float& v0, const float& v1, const float isoValue=0.f );
                                                            
    public:

        void initialize();

        void march( const FloatSparseField& field, Vec3fArray& vertices );
        void march( const ScalarDenseField& field, Vec3fArray& vertices );

};

BORA_NAMESPACE_END

#endif

