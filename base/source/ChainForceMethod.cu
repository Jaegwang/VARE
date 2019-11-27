//---------------------//
// ChainForceMethod.cu //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.08.23                               //
//-------------------------------------------------------//

#include <Bora.h>

void ChainForceMethod::
force( Vec3fArray& pos, Vec3fArray& vel, const float fScale, const float radius, const int iter )
{
    int num = KMeanClusterPoints( _centroids, _Ids, _sec, _table, pos, radius, iter );
    if( num < 5 ) return;

    Vec3fArray& centroids = _centroids;
    IndexArray& Ids = _Ids;
    IndexArray& sec = _sec;

    Vec3f* pPos = pos.pointer();
    Vec3f* pVel = vel.pointer();

    auto kernel = [=] BORA_DEVICE ( const size_t n )
    {
        size_t id = Ids[n];
        size_t se = sec[n];

        if( id == NULL_MAX || se == NULL_MAX ) return;

        const Vec3f& p = pos[n];

        const Vec3f& center = centroids[ id ];
        const Vec3f& target = centroids[ se ];
        
        const Vec3f dev = target-center;
        const float alpha = Clamp( ((p-center)*dev)/(dev*dev), 0.f, 1.f );
        
        Vec3f tar = center + dev*alpha;

        pVel[n] += (tar-p)* fScale;
    };

    LaunchCudaDevice( kernel, 0, pos.size() );
    SyncCuda();
}

