//----------------//
// KMeanCluster.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.04.02                               //
//-------------------------------------------------------//

#ifndef _BoraKMeanCluster_h_
#define _BoraKMeanCluster_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

int
KMeanClusterPoints( Vec3fArray& centroids, IndexArray& Ids, IndexArray& secIds, HashGridTable& table, const Vec3fArray& points, const float rad, const int maxIter );

BORA_NAMESPACE_END

#endif

