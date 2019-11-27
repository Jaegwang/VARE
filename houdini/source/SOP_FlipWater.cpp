//-------------------//
// SOP_FlipWater.cpp //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.11.23                               //
//-------------------------------------------------------//

#include <SOP_FlipWater.h>
#include <DM_FlipSimRender.h>
#include <DM_SimInfoRender.h>
#include <DM_VectorFieldRender.h>

const char* SOP_FlipWater::uniqueName = "borawater";
const char* SOP_FlipWater::labelName = "Bora Main Water";
const char* SOP_FlipWater::menuName = "Bora";

const char* SOP_FlipWater::inputLabels[] =
{
    "Simulation Box",
    "Source Points",
    "Collision Surface & Velocity",
    "Null"
};

PRM_Template SOP_FlipWater::nodeParameters[] =
{
    /**/
    houSwitcherParameter( "simprop", "simprop", 1, "Simulation Properties", 7 ),
    houIntParameter  ( "startframe" , "Start Frame", 1 ),
    houIntParameter  ( "substeps"   , "Max Sub Steps", 5 ),
    houFloatParameter( "timescale"  , "Time Scale", 1.f ),
    houFloatParameter( "colvelscale", "Collision Velocity Scale", 1.f ),
    houVec3fParameter( "gravity"    , "Gravity", Vec3f(0.f,-9.80665f,0.f) ),
    houFloatParameter( "vortpres"   , "Vorticity Preservation", 0.5f ),
    houBoolParameter ( "visinfo"    , "Visualize Sim Info", false ),
    /**/
    houSwitcherParameter( "controls", "controls", 2, "Water Tank", 5, "Curl Noise", 10 ),
    houBoolParameter ( "enabletank" , "Enable Water Tank", false ),
    houBoolParameter ( "fillwater"  , "Fill Water", false ),
    houFloatParameter( "waterlevel" , "Water Level", 0.f ),
    houIntParameter  ( "ptspercell" , "Particles per cell", 8 ),
    houFloatParameter( "dampingdist", "Damping Distance", 0.f ),
    houBoolParameter ( "addcurl"    , "Add Curl Noise", false ),
    houFloatParameter( "frequency"  , "Frequency", 1.f ),
    houVec3fParameter( "offset"     , "Offset", Vec3f(0.f) ),
    houFloatParameter( "scale"      , "Scale", 1.f ),
    houVec3fParameter( "addvelo"    , "Add Velocity", Vec3f(0.f) ),
    houFloatParameter( "adjunder"   , "Adjacency Under", 2.f ),
    houSeperatorParameter( "vissep" , "vissep" ),
    houBoolParameter ( "viscurl"    , "Visualize Curl", false ),
    houFloatParameter( "visdepth"   , "Depth", 0.5f ),
    houMenuParameter ( "visaxis"    , "Look Axis", 3, "X", "Y", "Z" ),
    /**/
    houSwitcherParameter( "advanced", "advanced", 1, "Advanced", 2 ),
    houIntParameter  ( "projection" , "Projection Iteration", 200 ),
    houIntParameter  ( "levelset"   , "Levelset Iteration", 80 ),
    /**/
    PRM_Template()
};

OP_Node*
SOP_FlipWater::constructor( OP_Network*net, const char *name, OP_Operator* entry )
{
    return new SOP_FlipWater( net, name, entry );
}

SOP_FlipWater::SOP_FlipWater( OP_Network*net, const char* name, OP_Operator* entry )
: SOP_Node(net, name, entry)
{
}

SOP_FlipWater::~SOP_FlipWater()
{
    DM_FlipSimRender::map.erase( size_t(this) );
    DM_VectorFieldRender::map.erase( size_t(this) );
    DM_SimInfoRender::map.erase( size_t(this ) );
}

OP_ERROR
SOP_FlipWater::cookMySop( OP_Context &context )
{
    OP_AutoLockInputs inputs( this );
    if (inputs.lock(context) >= UT_ERROR_ABORT) return error();

    OP_Node::flags().setTimeDep(1);

    CH_Manager *chman = OPgetDirector()->getChannelManager();

    const int fps = OPgetDirector()->getChannelManager()->getSamplesPerSec();
    const int currframe = chman->getSample(context.getTime());
    const int startframe = houEvalInt( this, "startframe" );


    Vec3f gridSize(10.f), gridCenter(0.f);
    float voxelSize(0.1f);

    const GU_Detail *boxgdp = inputGeo( 0 );
    if( boxgdp )
    {
        GA_ROHandleV3 rohSize = boxgdp->findFloatTuple( GA_ATTRIB_DETAIL, "gridsize", 3 );
        GA_ROHandleV3 rohCenter = boxgdp->findFloatTuple( GA_ATTRIB_DETAIL, "gridcenter", 3 );
        GA_ROHandleF rohVoxel = boxgdp->findFloatTuple( GA_ATTRIB_DETAIL, "voxelsize", 1 );

        if( rohSize.isValid() )
        {
            UT_Vector3 gSize = rohSize.get( GA_Offset(0) );
            gridSize = Vec3f( gSize.x(), gSize.y(), gSize.z() );
        }

        if( rohCenter.isValid() )
        {
            UT_Vector3 gCenter = rohCenter.get( GA_Offset(0) );
            gridCenter = Vec3f( gCenter.x(), gCenter.y(), gCenter.z() );
        }

        if( rohVoxel.isValid() ) voxelSize = rohVoxel.get( GA_Offset(0) );
    }

    Vec3f gridmin = gridCenter - (gridSize*0.5f);
    Vec3f gridmax = gridCenter + (gridSize*0.5f);

    _grid.initialize( voxelSize, gridmin, gridmax );

    _flip.params.dt = (1.f/(float)fps) * houEvalFloat( this, "timescale" );
    _flip.params.voxelSize       = voxelSize;
    _flip.params.ptsPerCell      = houEvalInt( this, "ptspercell" );
    _flip.params.colVelScale     = houEvalFloat( this, "colvelscale" );
    _flip.params.maxsubsteps     = houEvalInt( this, "substeps" );
    _flip.params.wallLevel       = houEvalFloat( this, "waterlevel" );
    _flip.params.extForce        = houEvalVec3f( this, "gravity" );
    _flip.params.vortPresRate    = houEvalFloat( this, "vortpres" );
    _flip.params.projIteration   = houEvalInt( this, "projection" );
    _flip.params.redistIteration = houEvalInt( this, "levelset" );
    _flip.params.enableWaterTank = houEvalBool( this, "enabletank" );
    _flip.params.dampingBand     = houEvalFloat( this, "dampingdist" );
    _flip.params.adjacencyUnder  = houEvalFloat( this, "adjunder" );
    _flip.params.enableCurl      = houEvalBool( this, "addcurl" );

    if( _flip.params.enableCurl )
    {
        _flip.curlNoise.scale     = houEvalFloat( this, "scale" );
        _flip.curlNoise.frequency = Vec4f( houEvalFloat( this, "frequency" ) );
        _flip.curlNoise.offset    = Vec4f( houEvalVec3f( this, "offset" ), 0.f );
        _flip.curlNoise.add       = houEvalVec3f( this, "addvelo" );
    }

    if( houEvalBool( this, "viscurl" ) == true )
    {
        DM_VectorFieldRender::Data data;
        data.grid = _grid;
        data.noise = _flip.curlNoise;
        data.axis = houEvalInt( this, "visaxis" );
        data.offset = houEvalFloat( this, "visdepth" );
        data.frame = (float)currframe;

        DM_VectorFieldRender::map[size_t(this)] = data;
    }
    else
    {
        DM_VectorFieldRender::map.erase( size_t(this) );
    }

    bool simulated = false;
   
    if( currframe == startframe )
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

    if( houEvalBool( this, "visinfo") == true )
    {
        {
            DM_FlipSimRender::Data data;
            data.grid = _flip.getGrid();
            data.params = _flip.params;

            DM_FlipSimRender::map[size_t(this)] = data;
        }

        {
            DM_SimInfoRender::Data data;
            data.title = this->getName();            
            data.grid = _flip.getGrid();
            data.flipParams = _flip.params;
            data.mainParticles = _pts.position.size();
            data.message = _flip.watch.printString();

            DM_SimInfoRender::map[size_t(this)] = data;
        }
    }
    else
    {
        DM_FlipSimRender::map.erase( size_t(this) );
        DM_SimInfoRender::map.erase( size_t(this) );
    }

    // Simulation Output to bgeo
    gdp->stashAll();

    const Particles& hpts = _pts;
    
    // Export particles.
    if( hpts.position.num() )
    {   
        GA_Offset startoff = gdp->appendPointBlock( hpts.position.size() );

        if( hpts.position.size() > 0 )
        {
            GA_RWHandleV3 poseHnd = GA_RWHandleV3( gdp->addFloatTuple( GA_ATTRIB_POINT, "P", 3 ) );
            poseHnd.setBlock( startoff, hpts.position.size(), (UT_Vector3*)hpts.position.pointer() );
            poseHnd->setTypeInfo( GA_TYPE_POINT );
        }

        if( hpts.velocity.size() > 0 )
        {
            GA_RWHandleV3 veloHnd = GA_RWHandleV3( gdp->addFloatTuple( GA_ATTRIB_POINT, "v", 3 ) );
            veloHnd.setBlock( startoff, hpts.velocity.size(), (UT_Vector3*)hpts.velocity.pointer() );
            veloHnd->setTypeInfo( GA_TYPE_VECTOR );
        }

        if( hpts.vorticity.size() > 0 )
        {
            GA_RWHandleF vortHnd = GA_RWHandleF( gdp->addFloatTuple( GA_ATTRIB_POINT, "vorticity", 1 ) );
            vortHnd.setBlock( startoff, hpts.vorticity.size(), hpts.vorticity.pointer() );
        }

        if( hpts.curvature.size() > 0 )
        {
            GA_RWHandleF curvHnd = GA_RWHandleF( gdp->addFloatTuple( GA_ATTRIB_POINT, "curvature", 1 ) );
            curvHnd.setBlock( startoff, hpts.vorticity.size(), hpts.curvature.pointer() );
        }

        if( hpts.adjacency.size() > 0 )
        {
            GA_RWHandleF adjhHnd = GA_RWHandleF( gdp->addFloatTuple( GA_ATTRIB_POINT, "adjacency", 1 ) );
            adjhHnd.setBlock( startoff, hpts.adjacency.size(), hpts.adjacency.pointer() );
        }       

        GA_PointGroup* flip_group = gdp->newPointGroup( "bora_flip" );
        flip_group->addAll();
    }


    // Export VDBs
    const FloatSparseField& surfaceField  = _flip.getSurfaceField();
    const Vec3fSparseField& velocityField = _flip.getVelocityField();

    if( surfaceField.size() > 0 && simulated == true )
    {
        openvdb::GridPtrVec grids;

        openvdb::FloatGrid::Ptr grid;
        convertSurfaceVDB( grid, surfaceField, _flip.params.enableWaterTank, _flip.params.wallLevel );
        grids.push_back( grid );
       
        openvdb::Vec3fGrid::Ptr vel;
        convertVelocityVDB( vel, velocityField );
        grids.push_back( vel );

        for( int n=0; n<grids.size(); ++n )
        {
            std::string name = grids[n]->getName();
            GU_PrimVDB* vdb = GU_PrimVDB::buildFromGrid( *gdp, grids[n], 0, name.c_str() );

            if     ( name == "density" ) vdb->setVisualization( GEO_VOLUMEVIS_SMOKE    , vdb->getVisIso(), vdb->getVisDensity() );
            else if( name == "surface" ) vdb->setVisualization( GEO_VOLUMEVIS_INVISIBLE, vdb->getVisIso(), vdb->getVisDensity() );
        }
    }

    gdp->destroyStashed();

    return error();
}

void SOP_FlipWater::initialize( OP_Context &context )
{
    _flip.initialize( _grid );
    _pts.clear();

    if( _flip.params.enableWaterTank == true )
    {
        if( houEvalBool( this, "fillwater" ) ) _flip.fillWaterTank( _pts );
    }

}

void SOP_FlipWater::idle( OP_Context &context )
{    
    const GU_Detail *ptsgdp = inputGeo( 1 );
    if( ptsgdp )
    {
        const size_t np = ptsgdp->getNumPoints();
        GA_Offset startoff = ptsgdp->pointOffset( 0 );

        GA_ROHandleV3 rohPos = ptsgdp->findFloatTuple( GA_ATTRIB_POINT, "P", 3 );
        GA_ROHandleV3 rohVel = ptsgdp->findFloatTuple( GA_ATTRIB_POINT, "v", 3 );

        Vec3fArray posTmp, velTmp;

        posTmp.initialize( np, kUnified );
        if( rohPos.isValid() ) rohPos.getBlock( startoff, np, (UT_Vector3*)posTmp.pointer() );

        velTmp.initialize( np, kUnified );
        if( rohVel.isValid() ) rohVel.getBlock( startoff, np, (UT_Vector3*)velTmp.pointer() );

        Grid grid = _flip.getGrid();

        for( size_t n=0; n<np; ++n )
        {
            const Vec3f& p = grid.voxelPoint( posTmp[n] );

            if( grid.inside( p, 0.5f ) == true )
            {
                _flip.newParticle( _pts, posTmp[n], velTmp[n] );
            }
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
            _flip.updateCollisionSource( collision, vel );
        }
    }

    _flip.advanceOneFrame( _pts );
}

