//--------------------//
// ChainForceMethod.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.04.02                               //
//-------------------------------------------------------//

#pragma once

#include <Bora.h>

BORA_NAMESPACE_BEGIN

class ChainForceMethod
{
    public:

        // for k-mean clustering
        Vec3fArray _centroids;
        IndexArray _Ids, _sec;
        
        // for searching cluster centroids
        HashGridTable _table;

    public:

        void force( Vec3fArray& pos, Vec3fArray& vel, const float fScale, const float radius, const int iter=30 );

};

BORA_NAMESPACE_END

