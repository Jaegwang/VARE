//----------------------//
// SOP_DiffuseWater.cpp //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2019.04.04                               //
//-------------------------------------------------------//

#include <SOP_DiffuseWater.h>
#include <DM_VectorFieldRender.h>

const char* SOP_DiffuseWater::uniqueName = "boradiffusewater";
const char* SOP_DiffuseWater::labelName = "Bora Sub Water";
const char* SOP_DiffuseWater::menuName = "Bora";

const char* SOP_DiffuseWater::inputLabels[] = 
{
    "Simulation Box",
    "Source Water Body",
    "Collision Surface & Velocity VDB",
    "Source Points"
};

PRM_Template SOP_DiffuseWater::nodeParameters[] =
{
    /**/
    houSwitcherParameter( "simprop", "simprop", 1, "Simulation Properties", 13 ),
    houIntParameter  ( "startframe"     , "Start frame", 1 ),
    houVec3fParameter( "gravity"        , "Gravity", Vec3f(0.f,-9.80665f,0.f) ),    
    houFloatParameter( "mindensity"      , "Min Density", 1.2f ),
    houMenuParameter ( "typemenu"       , "Types", 4, "Splash", "Spray", "Foam", "Bubble" ),
    houFloatParameter( "potentialscale" , "Emit Rate", 1.f ),
    houFloatParameter( "lifetime"       , "Life Time", 10.f ),
    houFloatParameter( "lifevariance"   , "Life Variance", 5.f ),
    houFloatParameter( "accelscale"     , "Acceleration Scale", 0.2f ),
    houFloatParameter( "numrate"        , "Replication Rate", 1.f ),
    houVec2fParameter( "curvatures"     , "Curvature Range", Vec2f(1.f, 2.f) ),
    houSeperatorParameter( "simvissep", "simvissep" ),
    houBoolParameter( "visgrid", "Visualize Sim Grid", false ),
    houBoolParameter( "visinfo", "Visualize Sim Info", false ),   
    /**/
    houSwitcherParameter( "types", "types", 4, "Splash", 1, "Spray", 4, "Foam", 1, "Bubble", 2 ),
    //splash
    houSeperatorParameter( "splashsep", "splashsep" ),
    //spray
    houFloatParameter( "vortexforce"     , "Vortex Force", 0.8f ),
    houFloatParameter( "disspwater"      , "Dissipation Water", 1.f ),
    houFloatParameter( "disspair"        , "Dissipation Air", 1.f ),
    houIntParameter  ( "projiteration"   , "Projection Iteration", 15 ),
    //foam
    houSeperatorParameter( "foamsep", "foamsep" ),
    //bubble
    houFloatParameter( "bouyancy", "Bouyancy", 0.5f ),
    houFloatParameter( "drageffect", "Drag Effect", 0.5f ),

    // etc
    houSwitcherParameter( "etc", "etc", 1, "Curl Noise", 9 ),
    houBoolParameter ( "addcurl"         , "Add Curl Noise", false ),
    houFloatParameter( "curlfrequency"   , "Frequency", 1.f ),
    houVec3fParameter( "curloffset"      , "Offet", Vec3f(0.f) ),
    houFloatParameter( "curlscale"       , "Scale", 1.f ),
    houVec3fParameter( "addvelo"         , "Add Velocity", Vec3f(0.f) ),
    houSeperatorParameter( "vissep", "vissep" ),
    houBoolParameter ( "viscurl"         , "Visualize Curl", false ),
    houFloatParameter( "visdepth"        , "Depth", 0.5f ),
    houMenuParameter ( "visaxis"         , "Look Axis", 3, "X", "Y", "Z" ),

    //end
    PRM_Template()
};

OP_Node*
SOP_DiffuseWater::constructor( OP_Network*net, const char *name, OP_Operator* entry )
{
    return new SOP_DiffuseWater( net, name, entry );
}

SOP_DiffuseWater::SOP_DiffuseWater( OP_Network*net, const char* name, OP_Operator* entry )
: SOP_Node(net, name, entry)
{
}

SOP_DiffuseWater::~SOP_DiffuseWater()
{
    DM_VectorFieldRender::map.erase( size_t(this) );
}

bool
SOP_DiffuseWater::updateParmsFlags()
{
    int type = houEvalInt( this, "typemenu" );

    if( type == 0 ) // Splash
    {

    }
    else if( type == 1 ) // Spray
    {

    }
    else if( type == 2 ) // Foam
    {

    }
    else if( type == 3 ) // Bubble
    {

    }

    return SOP_Node::updateParmsFlags();
}

OP_ERROR
SOP_DiffuseWater::cookMySop( OP_Context &context )
{
    OP_AutoLockInputs inputs( this );
    if (inputs.lock(context) >= UT_ERROR_ABORT) return error();

    OP_Node:flags().timeDep = 1;

    CH_Manager *chman = OPgetDirector()->getChannelManager();
    int currframe = chman->getSample(context.getTime());

    const int startFrame = houEvalInt( this, "startframe" );

    const GU_Detail *boxgdp = inputGeo( 0 );
    if( boxgdp )
    {
        GA_ROHandleV3 rohSize = boxgdp->findFloatTuple( GA_ATTRIB_DETAIL, "gridsize", 3 );
        GA_ROHandleV3 rohCenter = boxgdp->findFloatTuple( GA_ATTRIB_DETAIL, "gridcenter", 3 );
        GA_ROHandleF rohVoxel = boxgdp->findFloatTuple( GA_ATTRIB_DETAIL, "voxelsize", 1 );

        UT_Vector3 gSize = rohSize.get( GA_Offset(0) );
        UT_Vector3 gCenter = rohCenter.get( GA_Offset(0) );

        _gridCenter = Vec3f( gCenter.x(), gCenter.y(), gCenter.z() );
        _gridSize = Vec3f( gSize.x(), gSize.y(), gSize.z() );
        _voxelSize = rohVoxel.get( GA_Offset(0) );

        const Vec3f gridMin = _gridCenter-(_gridSize*0.5f);
        const Vec3f gridMax = _gridCenter+(_gridSize*0.5f);
        Grid grid;        
        grid.initialize( _voxelSize, gridMin, gridMax );

        if( houEvalBool( this, "viscurl" ) == true )
        {
            DM_VectorFieldRender::Data data;
            data.grid = grid;
            data.offset = houEvalFloat( this, "visdepth" );
            data.axis = houEvalInt( this, "visaxis" );
            data.noise.frequency = Vec4f( houEvalFloat( this, "curlfrequency" ) );
            data.noise.offset = Vec4f( houEvalVec3f( this, "curloffset" ), 0.f ); 
            data.noise.scale = houEvalFloat( this, "curlscale" );
            data.noise.add = houEvalVec3f( this, "addvelo" );

            DM_VectorFieldRender::map[ size_t(this) ] = data;
        }
        else
        {
            DM_VectorFieldRender::map.erase( size_t(this) );
        }

        if( houEvalBool( this, "visgrid" ) == true )
        {
        }
        else
        {
        }

        if( houEvalBool( this, "visinfo" ) == true )
        {
        }
        else
        {
        }
    }

    bool simulated = false;

    if( currframe == startFrame )
    { // initialize simulation.
        initialize( context );
        cookedFrame = currframe;
    }
    else if( currframe == cookedFrame+1 )
    { // do simulate.
        idle( context );
        cookedFrame = currframe;

        simulated = true;
    }

    gdp->stashAll();

    const Particles& pts = _diffWater.getParticles();

    // Export points.
    if( pts.position.size() )
    {
        GA_Offset first_offset = gdp->appendPointBlock( pts.position.size() );

        if( pts.position.size() > 0 )
        {
            GA_RWHandleV3 posHnd = GA_RWHandleV3( gdp->addFloatTuple( GA_ATTRIB_POINT, "P", 3 ) );
            posHnd.setBlock( first_offset, pts.position.size(), (UT_Vector3*)pts.position.pointer() );
            posHnd->setTypeInfo( GA_TYPE_POINT );
        }

        if( pts.velocity.size() > 0 )
        {
            GA_RWHandleV3 velHnd = GA_RWHandleV3( gdp->addFloatTuple( GA_ATTRIB_POINT, "v", 3 ) );
            velHnd.setBlock( first_offset, pts.velocity.size(), (UT_Vector3*)pts.velocity.pointer() );
            velHnd->setTypeInfo( GA_TYPE_VECTOR );
        }

        GA_PointGroup* diffuse_group = gdp->newPointGroup( "bora_diffuse" );
        diffuse_group->addAll();
    }

    // Export VDBs.
    if( houEvalBool( this, "typemenu" ) == 1 )
    {
        const ScalarDenseField& densityField = _diffWater.getDensityField();
        const VectorDenseField& velocityField = _diffWater.getVelocityField();

        openvdb::GridPtrVec grids;

        openvdb::FloatGrid::Ptr grid;
        openvdb::Vec3fGrid::Ptr velgrid;

        convertDensityAndVelocityVDB( grid, densityField, velgrid, velocityField );
        grids.push_back( grid );
        grids.push_back( velgrid );

        for( int n=0; n<grids.size(); ++n )
        {
            std::string name = grids[n]->getName();
            GU_PrimVDB* vdb = GU_PrimVDB::buildFromGrid( *gdp, grids[n], 0, name.c_str() );

            if     ( name == "density" ) vdb->setVisualization( GEO_VOLUMEVIS_SMOKE, vdb->getVisIso(), vdb->getVisDensity() );
            else if( name == "surface" ) vdb->setVisualization( GEO_VOLUMEVIS_ISO  , vdb->getVisIso(), vdb->getVisDensity() );
        }
    }

    gdp->destroyStashed();

    return error();
}

void
SOP_DiffuseWater::initialize( OP_Context& context )
{
    Vec3f gridMin = _gridCenter-(_gridSize*0.5f);
    Vec3f gridMax = _gridCenter+(_gridSize*0.5f);

    Grid grid;
    grid.initialize( _voxelSize, gridMin, gridMax );

    DiffuseWaterMethod::Params& prm = _diffWater.params;

    prm.simType = houEvalInt( this, "typemenu" );

    prm.curvatureRange = houEvalVec2f( this, "curvatures" );

    prm.voxelSize = _voxelSize;
    prm.disspWater = houEvalFloat( this, "disspwater" );
    prm.disspAir = houEvalFloat( this, "disspair" );
    prm.iteration = houEvalInt( this, "projiteration" );
    prm.minDensity = houEvalFloat( this, "mindensity" );
    prm.gravity = houEvalVec3f( this, "gravity" );
    prm.bouyancy = houEvalFloat( this, "bouyancy" );
    prm.vortexForce = houEvalFloat( this, "vortexforce" );
    prm.dragEffect = houEvalFloat( this, "drageffect" );

    prm.potentialScale = houEvalFloat( this, "potentialscale" );
    prm.lifeTime = houEvalFloat( this, "lifetime" );
    prm.lifeVariance = houEvalFloat( this, "lifevariance" );
    prm.accelScale = houEvalFloat( this, "accelscale" );
    prm.seedScale = houEvalFloat( this, "numrate" );

    if( houEvalBool( this, "addcurl" ) == true )
    {
        _diffWater.curlNoise.scale = houEvalFloat( this, "curlscale" );
        _diffWater.curlNoise.frequency = Vec4f( houEvalFloat( this, "curlfrequency" ) );
        _diffWater.curlNoise.offset = Vec4f( houEvalVec3f( this, "curloffset" ), 0.f );
        _diffWater.curlNoise.add = houEvalVec3f( this, "addvelo" );
    }
    else
    {
        _diffWater.curlNoise.scale = 0.f;
        _diffWater.curlNoise.add = Vec3f(0.f);
    }

    _diffWater.initialize( grid );
}

void
SOP_DiffuseWater::idle( OP_Context& context )
{
    const int fps = OPgetDirector()->getChannelManager()->getSamplesPerSec();
    const float dt = (1.f/(float)fps);

    CH_Manager *chman = OPgetDirector()->getChannelManager();
    int currframe = chman->getSample(context.getTime());
   
    openvdb::FloatGrid::ConstPtr surface = 0;
    openvdb::FloatGrid::ConstPtr collision = 0;
    openvdb::Vec3fGrid::ConstPtr vel = 0;

    const GU_Detail* fluid_gdp = inputGeo( 1 );
    if( fluid_gdp )
    {
        /* Get Surface and Vel VDB */
        const int numprim = fluid_gdp->getNumPrimitives();
        for( int n=0; n<numprim; ++n )
        {
            GA_Offset primoff = fluid_gdp->primitiveOffset( GA_Index(n) );
            const GEO_Primitive* prim = fluid_gdp->getGEOPrimitive( primoff );

            if( prim && prim->getTypeId() == GA_PRIMVDB )
            {
                const GEO_PrimVDB* vdb = (const GEO_PrimVDB*)prim;

                openvdb::GridBase::ConstPtr baseGrid = vdb->getGridPtr();

                if( surface == 0 && baseGrid->isType< openvdb::FloatGrid>() && baseGrid->getName() == "surface" )
                {
                    openvdb::FloatGrid::ConstPtr sdf = openvdb::gridConstPtrCast<openvdb::FloatGrid>( baseGrid );
                    surface = sdf;
                }

                if( vel == 0 && baseGrid->isType< openvdb::Vec3fGrid>() && baseGrid->getName() == "vel" )
                {
                    openvdb::Vec3fGrid::ConstPtr v = openvdb::gridConstPtrCast<openvdb::Vec3fGrid>( baseGrid );
                    vel = v;
                }
            }
        }

        /* get particles */
        const GA_PointGroup* flipGroup = fluid_gdp->findPointGroup( "bora_flip" );
        if( flipGroup )
        {
            //Particles pts;
            const size_t np = flipGroup->entries();
            const GA_Offset startoff = flipGroup->findOffsetAtGroupIndex(0);

            GA_ROHandleV3 rohPose = fluid_gdp->findFloatTuple( GA_ATTRIB_POINT, "P", 3 );
            GA_ROHandleV3 rohVelo = fluid_gdp->findFloatTuple( GA_ATTRIB_POINT, "v", 3 );

            if( _fluidpts.position.size() < np )
            {
                size_t num = size_t( (float)np*1.5f );
                _fluidpts.position.initialize( num, kUnified );
                _fluidpts.velocity.initialize( num, kUnified );
            }

            if( rohPose.isValid() )
            {
                rohPose.getBlock( startoff, np, (UT_Vector3*)_fluidpts.position.pointer() );
            }
            if( rohVelo.isValid() )
            {
                rohVelo.getBlock( startoff, np, (UT_Vector3*)_fluidpts.velocity.pointer() );
            }

            if( vel && surface )
            {
                _diffWater.updateFluidSource( surface, vel );
                _diffWater.updateFluidParticles( _fluidpts );

                int type = houEvalInt( this, "typemenu" );

                if( type == 0 ) _diffWater.generateSplashParticles( _fluidpts, np, _voxelSize, dt );

                if( type == 1 ) _diffWater.generateSprayParticles( _fluidpts, np, _voxelSize, dt );

                if( type == 2 ) _diffWater.generateFoamParticles( _fluidpts, np, _voxelSize, dt );

                if( type == 3 ) _diffWater.generateBubbleParticles( _fluidpts, np, _voxelSize, dt );
            }
            else
            {
                std::cout<< "No Surface or Velocity Field" << std::endl;
            }
        }
        else
        {
            std::cout<< "No FLIP Group" <<std::endl;
        }
    }

    const GU_Detail* colgdp = inputGeo( 2 );
    if( colgdp ) 
    {
        const int numprim = colgdp->getNumPrimitives();

        openvdb::FloatGrid::ConstPtr collision = 0;
        openvdb::Vec3fGrid::ConstPtr vel = 0;

        for( int n=0; n<numprim; ++n )
        {
            GA_Offset primoff = colgdp->primitiveOffset( GA_Index(n) );
            const GEO_Primitive* prim = colgdp->getGEOPrimitive( primoff );

            if( prim && prim->getTypeId() == GA_PRIMVDB )
            {
                const GEO_PrimVDB* vdb = (const GEO_PrimVDB*)prim;

                openvdb::GridBase::ConstPtr baseGrid = vdb->getGridPtr();

                if( baseGrid->isType<openvdb::FloatGrid>() ) // collsion
                {
                    collision = openvdb::gridConstPtrCast<openvdb::FloatGrid>( baseGrid );
                }
                else if( baseGrid->isType<openvdb::Vec3fGrid>() ) // velocity
                {
                    vel = openvdb::gridConstPtrCast<openvdb::Vec3fGrid>( baseGrid );
                }
            }
        }

        if( collision )
        {
            _diffWater.updateCollisionSource( collision, vel );
        }
    }

    const GU_Detail* srcgdp = inputGeo( 3 );
    if( srcgdp )
    {
        const size_t np = srcgdp->getNumPoints();
        GA_Offset startoff = srcgdp->pointOffset( 0 );

        GA_ROHandleV3 rohPos = srcgdp->findFloatTuple( GA_ATTRIB_POINT, "P", 3 );
        GA_ROHandleV3 rohVel = srcgdp->findFloatTuple( GA_ATTRIB_POINT, "v", 3 );

        Vec3fArray posTmp, velTmp;

        posTmp.initialize( np, kUnified );
        if( rohPos.isValid() ) rohPos.getBlock( startoff, np, (UT_Vector3*)posTmp.pointer() );

        velTmp.initialize( np, kUnified );
        if( rohVel.isValid() ) rohVel.getBlock( startoff, np, (UT_Vector3*)velTmp.pointer() );

        _diffWater.seedParticles( posTmp, velTmp );
    }

    _diffWater.advanceOneFrame( currframe, dt );
}

