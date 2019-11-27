//--------------//
// FLIPMethod.h //
//-------------------------------------------------------//
// author: Jaegwawng Lim @ Dexter Studios                //
// last update: 2019.03.13                               //
//-------------------------------------------------------//

#pragma once
#include <Bora.h>

BORA_NAMESPACE_BEGIN

class FLIPMethod
{
    private:

        Grid _grid;

        FluidProjection _projection;

        CharArray _removedPts;

        Vec3f _voxelExtForce;
        float _voxelWallLevel;

        bool _isPrepared = false;

        int _currentFrame = 0;
        
    public:

        /* for debugging */
        TimeWatch watch;

        /* VDB surface fields */
        CollisionSource _collisions;

        ////////////////////////////////////
        SparseFrame* _frame=0;
        IndexSparseField  _sMarkField;
        ScalarSparseField _sMassField;
        ScalarSparseField _sSurfField;
        ScalarSparseField _sPresField;
        VectorSparseField _sVeloField;
        VectorSparseField _sCompField;
        ////////////////////////////////////
        DenseField<char>  _scanField;

        Noise curlNoise;

        struct Params
        {
            Vec3f extForce=Vec3f(0.f,-9.80665f,0.f);
            int   maxsubsteps=1;
            float dt=1.f/24.f;
            float wallLevel=-1e+30f;
            bool  enableWaterTank=false;
            bool  enableCurl=false;
            float dampingBand=0.f;
            float voxelSize=0.1f;
            float colVelScale=1.f;
            int   ptsPerCell=8;
            float vortPresRate=0.2f;
            float adjacencyUnder=2.f;
            int   projIteration=200;
            int   redistIteration=80;
        } params;

    public:

        void prepareNextStep( Particles& particles );

        void updateParticles( Particles& particles, const float dt );

        void newParticle( Particles& pts, const Vec3f& p, const Vec3f& v )
        {
            if( pts.position.size() == 0 )
            {
                pts.position.initialize( 0, Bora::MemorySpace::kUnified );
                pts.velocity.initialize( 0, Bora::MemorySpace::kUnified );
                pts.vorticity.initialize( 0, Bora::MemorySpace::kUnified );
                pts.curvature.initialize( 0, Bora::MemorySpace::kUnified );
                pts.adjacency.initialize( 0, Bora::MemorySpace::kUnified );
                
                size_t reserveNum = _grid.nx()*_grid.nz()*5;

                pts.position.reserve( reserveNum );
                pts.velocity.reserve( reserveNum );
                pts.vorticity.reserve( reserveNum );
                pts.curvature.reserve( reserveNum );
                pts.adjacency.reserve( reserveNum );
            }

            pts.position.append( p );
            pts.velocity.append( v );
            pts.vorticity.append( 0.f );
            pts.curvature.append( 0.f );
            pts.adjacency.append( 0.f );
        }

    public:

        FLIPMethod();

        void initialize( const Grid& grid );

        void advanceOneFrame( Particles& particles );
        void advanceOneStep( Particles& particles, const float sub_dt );
        void postProcess( Particles& particles );
        
        void updateCollisionSource( const openvdb::FloatGrid::ConstPtr& collision, const openvdb::Vec3fGrid::ConstPtr& vel )
        {
            _collisions.update( collision, vel );
        }

        void particlesToWorld( Particles& particles );
        void particlesToVoxel( Particles& particles );

        void fillWaterTank( Particles& particles );

        Grid getGrid() const { return _grid; }

        const VectorSparseField& getVelocityField() { return _sVeloField; }
        const ScalarSparseField& getDensityField()  { return _sMassField; }
        const ScalarSparseField& getSurfaceField()  { return _sSurfField; }
};

BORA_NAMESPACE_END

