//-------------//
// Advection.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.02.09                               //
//-------------------------------------------------------//

#ifndef _BoraAdvection_h_
#define _BoraAdvection_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

void collideParticleBoundary( Vec3f& pos, Vec3f& v, size_t nx, size_t ny, size_t nz, const float dt );

void AdvectFLIParticles( Particles& pts, const VectorDenseField& vels, const ScalarDenseField& surf, const VectorDenseField& grad, const float dt );

void AdvectFLIParticles( Particles& pts, const Vec3fSparseField& vels, const VDBCollisions& collisions, const float dt );

BORA_NAMESPACE_END

#endif

