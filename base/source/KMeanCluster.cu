//-----------------//
// KMeanCluster.cu //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.11.15                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

// inner functions
void _findClosestCentroid( IndexArray& Ids, IndexArray& secIds, const HashGridTable& table, const Vec3fArray& points, const Vec3fArray& centroids );
void _updateCentroidPositions( Vec3fArray& centroids, IntArray& counts, const Vec3fArray& points, const IndexArray& Ids );


int
KMeanClusterPoints( Vec3fArray& centroids, IndexArray& Ids, IndexArray& secIds, HashGridTable& table, const Vec3fArray& points, const float rad, const int maxIter )
{
	const float voxelSize = rad*2.f;

	std::unordered_set<size_t> cluster_i;
	cluster_i.reserve( points.size()/2 );

	for( size_t n=0; n<points.size(); ++n )
	{
		const Vec3f& p = points[n];

		size_t i = (size_t)(p.x/voxelSize);
		size_t j = (size_t)(p.y/voxelSize);
		size_t k = (size_t)(p.z/voxelSize);

		i = ((i%10240)+10240)%10240;
		j = ((j%10240)+10240)%10240;
		k = ((k%10240)+10240)%10240;
		size_t hash = ((i*73856093)^(j*19349663)^(k*83492791)) % points.size();

		cluster_i.insert( hash );
	}

	size_t numClusters = cluster_i.size();

	centroids.initialize( numClusters, kUnified );

	Vec3fArray tempCentroids;
	tempCentroids.initialize( numClusters, kUnified );

	for( size_t n=0; n<numClusters; ++n )
	{
		size_t x = RandInt( n*451234, 0, points.size()-1 );
		centroids[n] = points[x];
	}

	table.initialize( voxelSize, centroids.size()*2 );
	table.build( centroids, tempCentroids );

	Ids.initialize( points.size(), kUnified );
	secIds.initialize( points.size(), kUnified );
	
	IntArray counts;
	counts.initialize( numClusters, kUnified );

	for( int i=0; i<maxIter+1; ++i )
	{
		_findClosestCentroid( Ids, secIds, table, points, centroids );
			
		_updateCentroidPositions( centroids, counts, points, Ids );

		table.build( centroids, tempCentroids );
	}

	_findClosestCentroid( Ids, secIds, table, points, centroids );

	return centroids.size();
}

void
_findClosestCentroid( IndexArray& Ids, IndexArray& secIds, const HashGridTable& table, const Vec3fArray& points, const Vec3fArray& centroids )
{
	const size_t numCen = centroids.size();

	size_t* pSec = secIds.pointer();
	size_t* pIds = Ids.pointer();

	const float rad = table.getCellSize() * sqrt( 2.f );
	
	auto kernel = [=] BORA_DEVICE ( const size_t n )
	{
		const Vec3f& p = points[n];

		pIds[n] = NULL_MAX;
		pSec[n] = NULL_MAX;

		int ci,cj,ck;
		table.getIndices( p, ci,cj,ck );

		float closestDist = 1e+30f;		

		for( int k=ck-1; k<=ck+1; ++k )
		for( int j=cj-1; j<=cj+1; ++j )
		for( int i=ci-1; i<=ci+1; ++i )
		{
            Idx2 range = table( i,j,k );
            if( range[1] == 0 ) continue;

            for( size_t x=range[0]; x<range[0]+range[1]; ++x )
            {
				const float d = centroids[ x ].distanceTo( p );
				if( d > rad ) continue;

				if( d <= closestDist )
				{
					closestDist = d;			
					pIds[n] = x;
				}
			}
		}

		float secDist = 1e+30f;

		for( int k=ck-1; k<=ck+1; ++k )
		for( int j=cj-1; j<=cj+1; ++j )
		for( int i=ci-1; i<=ci+1; ++i )
		{
            Idx2 range = table( i,j,k );
            if( range[1] == 0 ) continue;

            for( size_t x=range[0]; x<range[0]+range[1]; ++x )
            {
				const float d = centroids[x].distanceTo( p );
				if( d > rad ) continue;

				if( d <= secDist && d > closestDist )
				{
					secDist = d;
					pSec[n] = x;
				}
			}
		}

	};

	LaunchCudaDevice( kernel, 0, points.size() );
	SyncCuda();

}

void
_updateCentroidPositions( Vec3fArray& centroids, IntArray& counts, const Vec3fArray& points, const IndexArray& Ids )
{
	centroids.zeroize();
	counts.zeroize();

	for( size_t n=0; n<points.size(); ++n )
	{
		const Vec3f& p = points[n];
		const size_t& id = Ids[n];

		if( id == NULL_MAX ) continue;

		centroids[id] += p;
		counts[id] += 1;
	}

	for( size_t n=0; n<centroids.size(); ++n )
	{
		centroids[n] /= (float)counts[n];
	}
}

BORA_NAMESPACE_END

