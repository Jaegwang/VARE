//-----------------//
// Rasterization.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.10.18                               //
//-------------------------------------------------------//

#pragma once
#include <Bora.h>

BORA_NAMESPACE_BEGIN

class Rasterization
{
    public:

        static void
        PointsToVelField( ScalarDenseField& mass, VectorDenseField& vels, const Vec3fArray& p_pos, const Vec3fArray& p_vel );

        static void
        ParticlesToVelField( ScalarDenseField& mass, VectorDenseField& vels, const Particles& pts );

        static void
        PointsToCountField( ScalarDenseField& mass, const PointArray& pos );

        static void
        PointsToVelField( ScalarSparseField& mass, VectorSparseField& vels, const Vec3fArray& p_pos, const Vec3fArray& p_vel );

        static void
        ParticlesToVelField( ScalarSparseField& mass, VectorSparseField& vels, const Particles& pts );

        static void 
        PointsToCountField( ScalarSparseField& mass, const PointArray& pos );

        static void
        ParticlesToSurface( ScalarSparseField& surf, VectorSparseField& pose, const Particles& pts, const float influence=1.f, const float droplet=0.5f );

        static void
        ParticlesToSurfaceAndVelocity( ScalarSparseField& surf, ScalarSparseField& mass, VectorSparseField& pose, VectorSparseField& velo, const Particles& pts, const float influence=1.f, const float droplet=0.5f );

        static void
        ReinitializeSurface( ScalarSparseField& surf, ScalarSparseField& temp, const int iter, const bool watertank=false, const float waterlevel=-1e+10f );
};

BORA_NAMESPACE_END

