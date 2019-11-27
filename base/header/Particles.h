//-------------//
// Particles.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
//         Jaegwang Lim @ Dexter Studios                 //
// last update: 2019.02.20                               //
//-------------------------------------------------------//

#pragma once
#include <Bora.h>

BORA_NAMESPACE_BEGIN

class Particles
{
    public: // to be saved (attributes)

        AABB3f aabb;

        int   groupId;
        Vec3f groupColor;
        float timeScale;

        // long name
        CharArray   type, state;
        UIntArray   flags;
        IndexArray  uniqueId, index;
        FloatArray  radius, mass, density, adjacency, distance, age, lifespan, curvature, vorticity, velocityNormal;
        Vec3fArray  position, previous, velocity, curl, acceleration, force, angularMomentum, normal, color, textureCoordinates;
        QuatfArray  orientation;
        Mat33fArray deformationGradient;

        Vec3fArray  drawingColor;

        // sorting
        IndexArray sortIdx;

        // short name (reference)
        CharArray   &typ, &stt;
        IndexArray  &uid, &idx; 
        FloatArray  &rad, &mss, &den, &dst, /*age,*/ &lfs, &vrt;
        Vec3fArray  &pos, &prv, &vel, &acc, &frc, &amn, &nrm, &clr, &uvw, &dcr;
        QuatfArray  &ort;
        Mat33fArray &dgr;

    public:

        Particles( MemorySpace memorySpace=MemorySpace::kHost );

        void initialize( MemorySpace memorySpace=MemorySpace::kHost );

        Particles& operator=( const Particles& ptc );

        // Caustion) reset() and clear() are different.
        // reset() = clear() + alpha
        void reset();
        void clear();

        void updateBoundingBox();

        //void remove( const IndexArray& indicesToBeDeleted );
        void remove( const Array<char>& deleteMask );

        void draw() const;

		void write( std::ofstream& fout ) const;
		void read( std::ifstream& fin );

        bool save( const char* filePathName ) const;
        bool load( const char* filePathName );
};

std::ostream& operator<<( std::ostream& os, const Particles& object );

BORA_NAMESPACE_END

