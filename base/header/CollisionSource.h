//-------------------//
// CollisionSource.h //
//-------------------------------------------------------//
// author: Jaegwawng Lim @ Dexter Studios                //
// last update: 2019.03.13                               //
//-------------------------------------------------------//

#pragma once
#include <Bora.h>

BORA_NAMESPACE_BEGIN

class CollisionSource
{
    public:

        Grid _grid;
        float _voxelSize;

        bool _curv;

        SparseFrame* _frame=0;

        FloatSparseField _surfField;
        FloatSparseField _curvField;
        Vec3fSparseField _gradField;
        Vec3fSparseField _veloField;

        /* temp buffer */
        PointArray _voxelBuffer;

    public:

        void initialize( const Grid& inGrid, const bool useCurve=false );

        void update( const openvdb::FloatGrid::ConstPtr& collision, const openvdb::Vec3fGrid::ConstPtr& vel );

        BORA_FUNC_QUAL float signedDistance( const Vec3f& vPos ) const;
        BORA_FUNC_QUAL Vec3f normal( const Vec3f& vPos ) const;
        BORA_FUNC_QUAL Vec3f velocity( const Vec3f& vPos ) const;
        BORA_FUNC_QUAL float curvature( const Vec3f& vPos ) const;
};

inline void
CollisionSource::initialize( const Grid& inGrid, const bool useCurve )
{
    SparseFrame::remove( _frame );
    _frame = SparseFrame::create( inGrid, kUnified );
    
    _grid = _frame->_grid;
    _curv = useCurve;

    const float Lx = _grid.boundingBox().width(0);
    _voxelSize = Lx / (float)_grid.nx();

    _surfField.initialize( _frame, 5.f        );
    _gradField.initialize( _frame, Vec3f(0.f) );
    _veloField.initialize( _frame, Vec3f(0.f) );

    if( _curv ) _curvField.initialize( _frame, 0.f );

    _voxelBuffer.initialize( 0, kUnified );
    _voxelBuffer.reserve( _grid.nx()*_grid.nz() );
}

inline void
CollisionSource::update( const openvdb::FloatGrid::ConstPtr& collision, const openvdb::Vec3fGrid::ConstPtr& vel )
{
    if( collision == 0 ) return;

    typedef openvdb::FloatGrid GridType;
    _voxelBuffer.clear();

    GridType::ConstPtr grid = collision;
    openvdb::Vec3fGrid::Ptr grad = openvdb::tools::gradient( *collision );

    for (GridType::ValueOnCIter iter = grid->cbeginValueOn(); iter.test(); ++iter)
    {
        const float& value = (*iter);
        openvdb::Vec3d xyz = iter.getCoord().asVec3d();
        xyz = grid->transform().indexToWorld( xyz );

        Vec3f wp( xyz.x(), xyz.y(), xyz.z() );
        Vec3f vp = _grid.voxelPoint( wp );

        _voxelBuffer.append( vp );
    }

    const float h = _voxelSize;
    _surfField.background = collision->background() / h;
   
    _frame->buildFromPoints( _voxelBuffer );
    
    typedef openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> floatSampler;
    typedef openvdb::tools::GridSampler<openvdb::Vec3fGrid, openvdb::tools::BoxSampler> vec3fSampler;

    floatSampler surfSampler( *collision );
    vec3fSampler veloSampler( *vel );
    vec3fSampler gradSampler( *grad );

    auto kernel_collision = [&] BORA_HOST ( const size_t n )
    {
        size_t i,j,k;
        _surfField.findIndices( n,i,j,k );

        Vec3f wp = _surfField.worldPoint( _surfField.cellCenter( i,j,k ) );

        _surfField[n] = surfSampler.wsSample( openvdb::Vec3R( wp.x, wp.y, wp.z ) ) / h;

        if( vel )
        {
            openvdb::Vec3R v = veloSampler.wsSample( openvdb::Vec3R( wp.x, wp.y, wp.z ) );
            _veloField[n] = _veloField.voxelVector( Vec3f(v.x(), v.y(), v.z()) );
        }

        openvdb::Vec3R g = gradSampler.wsSample( openvdb::Vec3R( wp.x, wp.y, wp.z ) );

        Vec3f norm(g.x(), g.y(), g.z());

        _gradField[n] = _gradField.voxelVector( norm ).normalized();
    };

    LaunchCudaHost( kernel_collision, 0, _surfField.size() );

    if( _curv ) FieldOP::curvature( _gradField, _curvField );
}

inline float CollisionSource::signedDistance( const Vec3f& vPos ) const
{
    if( _surfField.size() == 0 ) return _surfField.background;
    return _surfField.lerp( vPos );
}

inline Vec3f CollisionSource::normal( const Vec3f& vPos ) const
{
    if( _gradField.size() == 0 ) return _gradField.background;
    return _gradField.lerp( vPos );
}

inline Vec3f CollisionSource::velocity( const Vec3f& vPos ) const
{
    if( _veloField.size() == 0 ) return _veloField.background;
    return _veloField.lerp( vPos );
}

inline float CollisionSource::curvature( const Vec3f& vPos ) const
{
    if( _curvField.size() == 0 ) return _curvField.background;
    return _curvField.lerp( vPos );
}

BORA_NAMESPACE_END

