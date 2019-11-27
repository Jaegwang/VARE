//-----------------------//
// OceanTileVertexData.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.04.23                               //
//-------------------------------------------------------//

#ifndef _BoraOceanTileVertexData_h_
#define _BoraOceanTileVertexData_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

class OceanTileVertexData
{
    private:

        int   _N = -1;      // grid resolution
        float _L = -1.f;    // geometric length
        float _h = 0.f;     // cell size

        float eps = EPSILON; // epsilon

        bool       _initialized  = false;
        OceanTile* _oceanTilePtr = nullptr;

        bool _toCalcStatistics = false;

    public:

        float minHeight;
        float maxHeight;

    public:

        Vec3fArray GRD; // grid vertex positions (before applying displacement)
        Vec3fArray POS; // mesh vertex positions (after applying displacement)
        Vec3fArray NRM; // mesh vertex normals
        FloatArray WAV; // mesh vertex wave values (height displacement)
        FloatArray CRS; // mesh vertex crest values
        UIntArray  TRI; // triangle connections

        // temporary arrays for interpolation
        // (to be activated only when oceanParams.flow > 0)
        Vec3fArray tmpPOS;
        Vec3fArray tmpNRM;
        FloatArray tmpWAV;
        FloatArray tmpCRS;

    public:

        OceanTileVertexData();

        void initialize( const OceanTile& oceanTile );

        void update( const float& time );

        const OceanTile& oceanTile() const;

        bool initialized() const;

        int vertexIndex( const int& i, const int& j ) const;

        int resolution() const;

        float tileSize() const;

        float cellSize() const;

        void drawWireframe( bool normals, bool crests ) const; // just for debugging

        void drawOutline() const;

        bool exportToEXR( const char* filePathName ) const;

        // Note) valid data: GRD, POS, CRS
        bool importFromEXR( const char* filePathName, const float L );

    public:

        void lerp( const Vec3f& worldPosition, Vec3f* grd, Vec3f* pos, Vec3f* nrm, Vec3f* wav, float* crs ) const;

        void catrom( const Vec3f& worldPosition, Vec3f* grd, Vec3f* pos, Vec3f* nrm, Vec3f* wav, float* crs ) const;

    private:

        void buildMesh( const int N, const float L );
};

BORA_NAMESPACE_END

#endif

