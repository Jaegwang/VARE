//----------------------//
// DiffuseWaterMethod.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2019.04.16                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

class DiffuseWaterMethod
{
    private:

        Grid _grid;
        SparseFrame* _grid_frame=0;

        /* simulaiton fields */
        VectorDenseField _veloField;
        VectorDenseField _compField;
        ScalarDenseField _massField;
        ScalarDenseField _diveField;
        ScalarDenseField _presField;
        ScalarDenseField _areaField;
        DenseField<char> _markField;

        /* diffuse particles */
        Particles _pts;
        CharArray _removedPts;

        /* properties */
        size_t _seedNum=1;        
        int _currentFrame=0;

        /* collision volumes */
        CollisionSource _fluidBody;
        CollisionSource _collisions;

    public:

        TimeWatch watch;

        /* control parameters */
        struct Params
        {
            int    simType=0; // 0:Splash, 1:Spray, 2:Foam, 3:Bubble
            float  voxelSize=0.1f;
            Vec3f  gravity=Vec3f(0.f,-9.8f,0.f);
            float  bouyancy=0.5f;
            float  dragEffect=0.5f; // bubble drag
            float  disspWater=1.f;
            float  disspAir=1.f;
            size_t iteration=10; // for projection
            float  vortexForce=0.8f; // vortex particle method

            float potentialScale=1.f;
            float accelScale=0.2f;
            float seedScale=1.f;
            float minDensity=1.2f;
            float lifeTime=10.f;
            float lifeVariance=10.f;

            float adjacencyUnder=2.f;
            float curvatureOver=2.f;
           
            Vec2f curvatureRange;
            Vec2f vorticityRange;
            Vec2f divergenceRange;
            Vec2f velocityRange;
            
        } params;

        Noise curlNoise;

    public:

        void initialize( const Grid& grid );

        void advanceOneFrame( const int currframe, const float dt );
        void advanceOneStep( const float dt );

        void generateSubParticles( const Vec3f& p, const Vec3f& v, const float voxel, const float dt, const size_t seed=1 );

        void generateSplashParticles( const Particles& fluid_pts, const size_t num, const float rad, const float dt );
        void generateSprayParticles( const Particles& fluid_pts, const size_t num, const float rad, const float dt );
        void generateBubbleParticles( const Particles& fluid_pts, const size_t num, const float rad, const float dt );
        void generateFoamParticles( const Particles& fluid_pts, const size_t num, const float rad, const float dt );

        void seedParticles( PointArray& pose, Vec3fArray& velo );

        void updateSplashParticles( const float dt );
        void updateSprayParticles( const float dt );
        void updateBubbleParticles( const float dt );
        void updateFoamParticles( const float dt );

        void removeParticles( const float dt );

        void updateFluidSource( const openvdb::FloatGrid::ConstPtr& sdf, const openvdb::Vec3fGrid::ConstPtr& velo );
        void updateCollisionSource( const openvdb::FloatGrid::ConstPtr& collision, const openvdb::Vec3fGrid::ConstPtr& vel );
        
        void updateFluidParticles( Particles& pts );

        const Particles& getParticles() { return _pts; }

        ScalarDenseField& getDensityField()  { return _massField; }
        VectorDenseField& getVelocityField() { return _veloField; }
};

inline void
DiffuseWaterMethod::updateFluidSource( const openvdb::FloatGrid::ConstPtr& sdf, const openvdb::Vec3fGrid::ConstPtr& vel )
{
    _fluidBody.update( sdf, vel );
}

inline void 
DiffuseWaterMethod::updateCollisionSource( const openvdb::FloatGrid::ConstPtr& collision, const openvdb::Vec3fGrid::ConstPtr& vel )
{
    _collisions.update( collision, vel );
}

BORA_NAMESPACE_END

