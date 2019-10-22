
#include <VARE.h>

VARE_NAMESPACE_BEGIN

SparseFrame* SparseFrame::create( const Grid& vGrid, MemorySpace mType )
{
    SparseFrame* newFrame=0;
    cudaMallocManaged( &newFrame, sizeof(SparseFrame) );
    *newFrame = SparseFrame();

    newFrame->initialize( vGrid, mType );
    
    return newFrame;
}

void SparseFrame::remove( SparseFrame* frame )
{
    if( frame ) cudaFree( frame );
}

void SparseFrame::initialize( const Grid& vGrid, MemorySpace mType )
{
    _memType = mType;

    const size_t block = _blockWidth;
    const size_t gx = vGrid.nx();
    const size_t gy = vGrid.ny();
    const size_t gz = vGrid.nz();

    _fx = (gx/block) + ((gx%block) ? 1 : 0);
    _fy = (gy/block) + ((gy%block) ? 1 : 0);
    _fz = (gz/block) + ((gz%block) ? 1 : 0);

    const size_t nx = _fx * block;
    const size_t ny = _fy * block;
    const size_t nz = _fz * block;

    Vec3f minP = vGrid.boundingBox().minPoint();
    Vec3f maxP = vGrid.boundingBox().maxPoint();
    Vec3f cenP = (minP + maxP)*0.5f;

    const float dx = (maxP.x-minP.x)/(float)gx;
    const float dy = (maxP.y-minP.y)/(float)gy;
    const float dz = (maxP.z-minP.z)/(float)gz;
    const float h = Min( Min(dx,dy),dz );

    const float lx = (float)nx * h;
    const float ly = (float)ny * h;
    const float lz = (float)nz * h;

    minP = cenP - Vec3f( lx*0.5f, ly*0.5f, lz*0.5f );
    maxP = minP + Vec3f( lx, ly, lz );

    size_t num = _fx*_fy*_fz;

	_fxfy = _fx*_fy;
	_br = _blockWidth;
	_brbr = _br*_br;
	_brbrbr = _br*_br*_br;

    _pointers.initialize( num, kUnified );
    _pointers.setValueAll( INVALID_MAX );

    _markers.initialize( num, kUnified );
    _markers.setValueAll( 0 );

    _coords.initialize( 0, kUnified );
    _coords.reserve( num/block+1 );

    _grid.initialize( nx, ny, nz, minP, maxP );
}

void SparseFrame::build()
{
    for( auto it=_map.begin(); it!=_map.end(); ++it )
    {
        it->second->build();
    }
}

void SparseFrame::buildFromPoints( const PointArray& points, const bool enableTank, const float height )
{
    const int fx = _fx;
    const int fy = _fy;
    const int fz = _fz;
    const int block_res = _blockWidth;
    const int block_size = _blockWidth*_blockWidth*_blockWidth;

    const float hh = (float)block_res*0.5f;

    _markers.zeroize();
    int* pMark = _markers.pointer();

    auto kernel = VARE_DEVICE_KERNEL
    {
        const Vec3f& p = points[ix];
        
        int ci = Max( (int)(p.x)/block_res, 0 );
        int cj = Max( (int)(p.y)/block_res, 0 );
        int ck = Max( (int)(p.z)/block_res, 0 );

        for( int k=ck-1; k<=Min(ck+1,fz-1); ++k )
        for( int j=cj-1; j<=Min(cj+1,fy-1); ++j )
        for( int i=ci-1; i<=Min(ci+1,fx-1); ++i )
        {
            size_t t = k*fx*fy + j*fx + i;

            int& m = pMark[t];
            atomicAdd( &m, 1 );
        }
    };

    LaunchDeviceKernel( kernel, 0, points.size() );
    SyncKernels();

    if( enableTank == true )
    {
        const int tall = Clamp( (int)(height)/block_res, 0, (int)_fy );
        for( int k=0; k<_fz  ; ++k )
        for( int j=0; j< tall; ++j )
        for( int i=0; i<_fx  ; ++i )
        {
            size_t t = k*fx*fy + j*fx + i;
            pMark[t] = 1;
        }
    }

    _coords.clear();

    size_t count = 0;
    for( size_t n=0; n<_markers.size(); ++n )
    {
        if( _markers[n] > 0 )
        {
            _pointers[n] = count;

            Idx3 coord;
            coord.i = (n) % _fx;
            coord.j = (n/_fx) % _fy;
            coord.k = (n/(_fx*_fy)) % _fz;
            _coords.append( coord );

            count += block_size;
        }
        else
        {
            _pointers[n] = INVALID_MAX;
        }
    }

    build();
}

VARE_NAMESPACE_END

