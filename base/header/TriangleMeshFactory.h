//-----------------------//
// TriangleMeshFactory.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.01.16                               //
//-------------------------------------------------------//

#ifndef _BoraTriangleMeshFactory_h_
#define _BoraTriangleMeshFactory_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

void CreatePlane( size_t Nx, size_t Ny, float Lx, float Ly, Axis axis, TriangleMesh* o_meshPtr );

BORA_NAMESPACE_END

#endif

