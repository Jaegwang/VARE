
#pragma once
#include <VARE.h>

VARE_NAMESPACE_BEGIN

inline void
convertVelocityVDB( openvdb::Vec3fGrid::Ptr& grid, const Vec3fSparseField& vel )
{
    const Vec3f min = vel.minPoint();
    const float dx = vel.dx();
    const float dy = vel.dy();
    const float dz = vel.dz();

    openvdb::Vec3d translate( min.x+dx*0.5f, min.y+dy*0.5f, min.z+dz*0.5f );

    openvdb::math::Transform::Ptr transform = openvdb::math::Transform::createLinearTransform( dx );
    transform->postTranslate( translate );

    grid = openvdb::Vec3fGrid::create( openvdb::Vec3f(0.f) );

    grid->setTransform( transform );
    grid->setName( "vel" );

    openvdb::Vec3fGrid::Accessor vel_acc = grid->getAccessor();

    for( size_t n=0; n<vel.size(); ++n )
    {
        const Vec3f v = vel.worldVector( vel[n] );
        if( v.length() < 1e-30f ) continue;        

        size_t i,j,k;
        vel.findIndices( n, i,j,k );

        openvdb::Vec3f vdb_v( v.x, v.y, v.z );

        openvdb::Coord idx( i,j,k );

        vel_acc.setValue( idx, vdb_v );
    }
}

inline void
convertVelocityVDB( openvdb::Vec3fGrid::Ptr& grid, const Vec3fSparseField& vel, const ScalarSparseField& sdf, const float band )
{
    const Vec3f min = vel.minPoint();
    const float dx = vel.dx();
    const float dy = vel.dy();
    const float dz = vel.dz();

    openvdb::Vec3d translate( min.x+dx*0.5f, min.y+dy*0.5f, min.z+dz*0.5f );

    openvdb::math::Transform::Ptr transform = openvdb::math::Transform::createLinearTransform( dx );
    transform->postTranslate( translate );

    grid = openvdb::Vec3fGrid::create( openvdb::Vec3f(0.f) );

    grid->setTransform( transform );
    grid->setName( "vel" );

    openvdb::Vec3fGrid::Accessor vel_acc = grid->getAccessor();

    for( size_t n=0; n<vel.size(); ++n )
    {
        if( Abs(sdf[n]) > band ) continue;

        const Vec3f v = vel.worldVector( vel[n] );
        if( v.length() < 1e-30f ) continue;

        size_t i,j,k;
        vel.findIndices( n, i,j,k );

        openvdb::Vec3f vdb_v( v.x, v.y, v.z );

        openvdb::Coord idx( i,j,k );

        vel_acc.setValue( idx, vdb_v );
    }
}

inline void
convertVelocityVDB( openvdb::Vec3fGrid::Ptr& grid, const VectorDenseField& vel )
{
    const Vec3f min = vel.minPoint();
    const float dx = vel.dx();
    const float dy = vel.dy();
    const float dz = vel.dz();

    openvdb::Vec3d translate( min.x+dx*0.5f, min.y+dy*0.5f, min.z+dz*0.5f );

    openvdb::math::Transform::Ptr transform = openvdb::math::Transform::createLinearTransform( dx );
    transform->postTranslate( translate );

    grid = openvdb::Vec3fGrid::create( openvdb::Vec3f(0.f) );

    grid->setTransform( transform );
    grid->setName( "vel" );

    openvdb::Vec3fGrid::Accessor vel_acc = grid->getAccessor();

    for( size_t n=0; n<vel.size(); ++n )
    {
        const Vec3f v = vel.worldVector( vel[n] );
        if( v.length() < 1e-30f ) continue;

        size_t i,j,k;
        vel.cellIndices( n, i,j,k );

        openvdb::Vec3f vdb_v( v.x, v.y, v.z );

        openvdb::Coord idx( i,j,k );

        vel_acc.setValue( idx, vdb_v );
    }
}

inline void
convertVelocityVDB( openvdb::FloatGrid::Ptr& grid_x, openvdb::FloatGrid::Ptr& grid_y, openvdb::FloatGrid::Ptr& grid_z, const VectorDenseField& vel )
{
    const Vec3f min = vel.minPoint();
    const float dx = vel.dx();
    const float dy = vel.dy();
    const float dz = vel.dz();

    openvdb::Vec3d translate( min.x+dx*0.5f, min.y+dy*0.5f, min.z+dz*0.5f );

    openvdb::math::Transform::Ptr transform = openvdb::math::Transform::createLinearTransform( dx );
    transform->postTranslate( translate );

    grid_x = openvdb::FloatGrid::create( 0.f );
    grid_y = openvdb::FloatGrid::create( 0.f );
    grid_z = openvdb::FloatGrid::create( 0.f );

    grid_x->setTransform( transform );
    grid_x->setName( "vel.x" );
    grid_y->setTransform( transform );
    grid_y->setName( "vel.y" );
    grid_z->setTransform( transform );
    grid_z->setName( "vel.z" );

    openvdb::FloatGrid::Accessor velx_acc = grid_x->getAccessor();
    openvdb::FloatGrid::Accessor vely_acc = grid_y->getAccessor();
    openvdb::FloatGrid::Accessor velz_acc = grid_z->getAccessor();

    for( size_t n=0; n<vel.size(); ++n )
    {
        const Vec3f v = vel.worldVector( vel[n] );
        if( v.length() < 1e-30f ) continue;

        size_t i,j,k;
        vel.cellIndices( n, i,j,k );

        openvdb::Coord idx( i,j,k );

        velx_acc.setValue( idx, v.x );
        vely_acc.setValue( idx, v.y );
        velz_acc.setValue( idx, v.z );
    }
}

inline void
convertVelocityVDB( openvdb::FloatGrid::Ptr& grid_x, openvdb::FloatGrid::Ptr& grid_y, openvdb::FloatGrid::Ptr& grid_z, const Vec3fSparseField& vel )
{
    const Vec3f min = vel.minPoint();
    const float dx = vel.dx();
    const float dy = vel.dy();
    const float dz = vel.dz();

    openvdb::Vec3d translate( min.x+dx*0.5f, min.y+dy*0.5f, min.z+dz*0.5f );

    openvdb::math::Transform::Ptr transform = openvdb::math::Transform::createLinearTransform( dx );
    transform->postTranslate( translate );

    grid_x = openvdb::FloatGrid::create( 0.f );
    grid_y = openvdb::FloatGrid::create( 0.f );
    grid_z = openvdb::FloatGrid::create( 0.f );

    grid_x->setTransform( transform );
    grid_x->setName( "vel.x" );
    grid_y->setTransform( transform );
    grid_y->setName( "vel.y" );
    grid_z->setTransform( transform );
    grid_z->setName( "vel.z" );

    openvdb::FloatGrid::Accessor velx_acc = grid_x->getAccessor();
    openvdb::FloatGrid::Accessor vely_acc = grid_y->getAccessor();
    openvdb::FloatGrid::Accessor velz_acc = grid_z->getAccessor();

    for( size_t n=0; n<vel.size(); ++n )
    {
        const Vec3f v = vel.worldVector( vel[n] );
        if( v.length() < 1e-30f ) continue;

        size_t i,j,k;
        vel.findIndices( n, i,j,k );
       
        openvdb::Coord idx( i,j,k );

        velx_acc.setValue( idx, v.x );
        vely_acc.setValue( idx, v.y );
        velz_acc.setValue( idx, v.z );
    }
}

inline void 
convertDensityVDB( openvdb::FloatGrid::Ptr& grid, const FloatSparseField& den )
{
    const Vec3f min = den.minPoint();
    const float dx = den.dx();
    const float dy = den.dy();
    const float dz = den.dz();

    openvdb::Vec3d translate( min.x+dx*0.5f, min.y+dy*0.5f, min.z+dz*0.5f );

    openvdb::math::Transform::Ptr transform = openvdb::math::Transform::createLinearTransform( dx );
    transform->postTranslate( translate );

    grid = openvdb::FloatGrid::create( 0.f );
    grid->setTransform( transform );
    grid->setName( "density" );
    grid->setGridClass( openvdb::GRID_FOG_VOLUME );

    openvdb::FloatGrid::Accessor acc = grid->getAccessor();

    for( size_t n=0; n<den.size(); ++n )
    {
        const float d = den[n];
        if( d < 1e-30f ) continue;

        size_t i,j,k;
        den.findIndices( n, i,j,k );

        openvdb::Coord idx( i, j, k );
        acc.setValue( idx, d );
    }    
}

inline void 
convertDensityAndVelocityVDB( openvdb::FloatGrid::Ptr& grid, const FloatSparseField& den, openvdb::Vec3fGrid::Ptr& velo_grid, const Vec3fSparseField& velo )
{
    const Vec3f min = den.minPoint();
    const float dx = den.dx();
    const float dy = den.dy();
    const float dz = den.dz();

    openvdb::Vec3d translate( min.x+dx*0.5f, min.y+dy*0.5f, min.z+dz*0.5f );

    openvdb::math::Transform::Ptr transform = openvdb::math::Transform::createLinearTransform( dx );
    transform->postTranslate( translate );

    grid = openvdb::FloatGrid::create( 0.f );
    grid->setTransform( transform );
    grid->setName( "density" );
    grid->setGridClass( openvdb::GRID_FOG_VOLUME );

    velo_grid = openvdb::Vec3fGrid::create( openvdb::Vec3f(0.f) );
    velo_grid->setTransform( transform );
    velo_grid->setName( "vel" );

    openvdb::FloatGrid::Accessor d_acc = grid->getAccessor();
    openvdb::Vec3fGrid::Accessor v_acc = velo_grid->getAccessor();

    for( size_t n=0; n<den.size(); ++n )
    {
        const float& d = den[n];
        if( d < 1e-30f ) continue;

        const Vec3f& v = velo[n];
        openvdb::Vec3f vdb_v( v.x, v.y, v.z );        

        size_t i,j,k;
        den.findIndices( n, i,j,k );
        openvdb::Coord idx( i, j, k );

        d_acc.setValue( idx, d );
        v_acc.setValue( idx, vdb_v );
    }    
}

inline void 
convertDensityVDB( openvdb::FloatGrid::Ptr& grid, const ScalarDenseField& den )
{
    const Vec3f min = den.minPoint();
    const float dx = den.dx();
    const float dy = den.dy();
    const float dz = den.dz();

    openvdb::Vec3d translate( min.x+dx*0.5f, min.y+dy*0.5f, min.z+dz*0.5f );

    openvdb::math::Transform::Ptr transform = openvdb::math::Transform::createLinearTransform( dx );
    transform->postTranslate( translate );

    grid = openvdb::FloatGrid::create( 0.f );
    grid->setTransform( transform );
    grid->setName( "density" );
    grid->setGridClass( openvdb::GRID_FOG_VOLUME );

    openvdb::FloatGrid::Accessor acc = grid->getAccessor();

    for( size_t n=0; n<den.size(); ++n )
    {
        const float d = den[n];
        if( d < 1e-30f ) continue;

        size_t i,j,k;
        den.cellIndices( n, i,j,k );

        openvdb::Coord idx( i, j, k );
        acc.setValue( idx, d );
    }    
}

inline void 
convertDensityAndVelocityVDB( openvdb::FloatGrid::Ptr& grid, const ScalarDenseField& den, openvdb::Vec3fGrid::Ptr& velo_grid, const VectorDenseField& velo )
{
    const Vec3f min = den.minPoint();
    const float dx = den.dx();
    const float dy = den.dy();
    const float dz = den.dz();

    openvdb::Vec3d translate( min.x+dx*0.5f, min.y+dy*0.5f, min.z+dz*0.5f );

    openvdb::math::Transform::Ptr transform = openvdb::math::Transform::createLinearTransform( dx );
    transform->postTranslate( translate );

    grid = openvdb::FloatGrid::create( 0.f );
    grid->setTransform( transform );
    grid->setName( "density" );
    grid->setGridClass( openvdb::GRID_FOG_VOLUME );

    velo_grid = openvdb::Vec3fGrid::create( openvdb::Vec3f(0.f) );
    velo_grid->setTransform( transform );
    velo_grid->setName( "vel" );

    openvdb::FloatGrid::Accessor d_acc = grid->getAccessor();
    openvdb::Vec3fGrid::Accessor v_acc = velo_grid->getAccessor();    

    for( size_t n=0; n<den.size(); ++n )
    {
        const float d = den[n];
        if( d < 1e-30f ) continue;

        const Vec3f& v = velo[n];
        openvdb::Vec3f vdb_v( v.x, v.y, v.z );           

        size_t i,j,k;
        den.cellIndices( n, i,j,k );

        openvdb::Coord idx( i, j, k );
        d_acc.setValue( idx, d );
        v_acc.setValue( idx, vdb_v );
    }    
}

inline void 
convertSurfaceVDB( openvdb::FloatGrid::Ptr& grid, const FloatSparseField& sur, const bool watertank=false, const float waterlevel=-1e+10f )
{
    const Vec3f min = sur.minPoint();
    const float dx = sur.dx();
    const float dy = sur.dy();
    const float dz = sur.dz();

    openvdb::Vec3d translate( min.x+dx*0.5f, min.y+dy*0.5f, min.z+dz*0.5f );

    openvdb::math::Transform::Ptr transform = openvdb::math::Transform::createLinearTransform( dx );
    transform->postTranslate( translate );

    grid = openvdb::FloatGrid::create( sur.background*dx );

    grid->setName( "surface" );
    grid->setGridClass( openvdb::GRID_LEVEL_SET );
    grid->setTransform( transform );

    openvdb::FloatGrid::Accessor acc = grid->getAccessor();

    for( size_t n=0; n<sur.size(); ++n )
    {
        const float phi = sur[n]*dx;

        size_t i,j,k;
        sur.findIndices( n, i,j,k );

        acc.setValue( openvdb::Coord( i,j,k ), phi );
    }

    if( watertank == true )
    {
        int EI(sur.nx()-1), EJ(sur.ny()-1), EK(sur.nz()-1);
        int bnd = 3;
        int lv = (waterlevel-min.y)/dx + 1;

        for( int k=-bnd; k<=EK+bnd; ++k )
        for( int j=-bnd; j<=0; ++j )
        for( int i=-bnd; i<=EI+bnd; ++i )
        {
            const Vec3f cellCenter = sur.worldPoint( Vec3f( i+0.5f, j+0.5f, k+0.5f ) );
            acc.setValue( openvdb::Coord( i,j,k ), Min( cellCenter.y-waterlevel, acc.getValue( openvdb::Coord(i,j,k) ) ) );
        }

        for( int k=-bnd; k<=EK+bnd; ++k )
        for( int j=-bnd; j<=(int)lv+bnd; ++j )
        for( int i=-bnd; i<=0; ++i )
        {
            const Vec3f cellCenter = sur.worldPoint( Vec3f( i+0.5f, j+0.5f, k+0.5f ) );
            acc.setValue( openvdb::Coord( i,j,k ), Min( cellCenter.y-waterlevel, acc.getValue( openvdb::Coord(i,j,k) ) ) );
        }

        for( int k=-bnd; k<=EK+bnd; ++k )
        for( int j=-bnd; j<=(int)lv+bnd; ++j )
        for( int i=EI; i<=EI+bnd; ++i )
        {
            const Vec3f cellCenter = sur.worldPoint( Vec3f( i+0.5f, j+0.5f, k+0.5f ) );
            acc.setValue( openvdb::Coord( i,j,k ), Min( cellCenter.y-waterlevel, acc.getValue( openvdb::Coord(i,j,k) ) ) );
        }

        for( int k=-bnd; k<=0; ++k )
        for( int j=-bnd; j<=(int)lv+bnd; ++j )
        for( int i=-bnd; i<=EI+bnd; ++i )
        {
            const Vec3f cellCenter = sur.worldPoint( Vec3f( i+0.5f, j+0.5f, k+0.5f ) );
            acc.setValue( openvdb::Coord( i,j,k ), Min( cellCenter.y-waterlevel, acc.getValue( openvdb::Coord(i,j,k) ) ) );
        }

        for( int k=EK; k<=EK+bnd; ++k )
        for( int j=-bnd; j<=(int)lv+bnd; ++j )
        for( int i=-bnd; i<=EI+bnd; ++i )
        {
            const Vec3f cellCenter = sur.worldPoint( Vec3f( i+0.5f, j+0.5f, k+0.5f ) );
            acc.setValue( openvdb::Coord( i,j,k ), Min( cellCenter.y-waterlevel, acc.getValue( openvdb::Coord(i,j,k) ) ) );
        }
    }
}

inline void
VDBtoSparseSurfaceField( const std::vector<openvdb::FloatGrid::ConstPtr>& vdbs, ScalarSparseField& surf )
{
    typedef openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> sampler;
    std::vector< sampler > samplers;

    float back_phi(0.f);
    for( int n=0; n<vdbs.size(); ++n )
    {
        samplers.push_back( sampler( *vdbs[n] ) );
        back_phi = Max( back_phi, vdbs[n]->background() );
    }

    surf.background = back_phi;
    surf.setValueAll( back_phi );

    auto kernel = VARE_HOST_KERNEL
    {
        size_t i,j,k;
        surf.findIndices( ix,i,j,k );

        Vec3f wp = surf.cellCenter( i,j,k );
        wp = surf.worldPoint( wp );

        for( int x=0; x<samplers.size(); ++x )
        {
            float& phi = surf[ix];
            const float vdb_phi = samplers[x].wsSample( openvdb::Vec3R( wp.x, wp.y, wp.z ) );

            phi = Min( vdb_phi, phi );
        }
    };

    LaunchHostKernel( kernel, 0, surf.size() );
	SyncKernels();
}

inline void
VDBtoDenseSurfaceField( const std::vector<openvdb::FloatGrid::ConstPtr>& vdbs, ScalarDenseField& surf )
{
    typedef openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> sampler;
    std::vector< sampler > samplers;

    for( int n=0; n<vdbs.size(); ++n ) samplers.push_back( sampler( *vdbs[n] ) );

    auto kernel = VARE_HOST_KERNEL
    {
        size_t i,j,k;
        surf.cellIndices( ix, i,j,k );

        Vec3f wp = surf.cellCenter( i,j,k );
        wp = surf.worldPoint( wp );

        for( int x=0; x<samplers.size(); ++x )
        {
            float& phi = surf[ix];
            const float vdb_phi = samplers[x].wsSample( openvdb::Vec3R( wp.x, wp.y, wp.z ) );

            phi = Min( vdb_phi, phi );
        }
    };

    LaunchHostKernel( kernel, 0, surf.size() );
	SyncKernels();
}

VARE_NAMESPACE_END

