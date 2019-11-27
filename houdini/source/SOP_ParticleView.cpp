//----------------------//
// SOP_ParticleView.cpp //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2019.04.16                               //
//-------------------------------------------------------//

#include <SOP_ParticleView.h>
#include <DM_ParticleRender.h>

const char* SOP_ParticleView::uniqueName = "boraparticleview";
const char* SOP_ParticleView::labelName = "Bora Particle View";
const char* SOP_ParticleView::menuName = "Bora";

const char* SOP_ParticleView::inputLabels[] = 
{
    "Flip Points"
};

PRM_Template SOP_ParticleView::nodeParameters[] =
{
    houMenuParameter ( "typemenu", "Types", 2, "Adjacency", "Curvature" ),
    houFloatParameter( "adja", "Adjacency Under", 2.f ),
    houFloatParameter( "curv", "Curvature Over", 1.f ),
    houRGBParameter  ( "color", "Color", Vec3f( 1.f, 1.f, 1.f ) ),
    PRM_Template()
};

OP_Node*
SOP_ParticleView::constructor( OP_Network* net, const char* name, OP_Operator* entry )
{
    return new SOP_ParticleView( net, name, entry );
}

SOP_ParticleView::SOP_ParticleView( OP_Network* net, const char* name, OP_Operator* entry )
: SOP_Node(net, name, entry)
{
}

SOP_ParticleView::~SOP_ParticleView()
{
    DM_ParticleRender::map.erase( size_t(this) );
}

bool
SOP_ParticleView::updateParmsFlags()
{
    int type = houEvalInt( this, "typemenu" );

    if( type == 0 )
    {
        setVisibleState( "adja", true );
        setVisibleState( "curv", false ); 
    }
    if( type == 1 )
    {
        setVisibleState( "adja", false );
        setVisibleState( "curv", true );
    }

    return SOP_Node::updateParmsFlags();
}

OP_ERROR
SOP_ParticleView::cookMySop( OP_Context &context )
{
    OP_AutoLockInputs inputs( this );
    if (inputs.lock(context) >= UT_ERROR_ABORT) return error();

    fpreal t = context.getTime();
    int frame = context.getFrame();

    const GU_Detail* fluid_gdp = inputGeo( 0 );
    if( fluid_gdp )
    {
        /* get particles */
        const GA_PointGroup* flipGroup = fluid_gdp->findPointGroup( "bora_flip" );
        if( flipGroup )
        {
            const size_t np = flipGroup->entries();
            const GA_Offset startoff = flipGroup->findOffsetAtGroupIndex(0);

            GA_ROHandleV3 rohPos = fluid_gdp->findFloatTuple( GA_ATTRIB_POINT, "P", 3 );
            GA_ROHandleF  rohAdj = fluid_gdp->findFloatTuple( GA_ATTRIB_POINT, "adjacency", 1 );
            GA_ROHandleF  rohVor = fluid_gdp->findFloatTuple( GA_ATTRIB_POINT, "vorticity", 1 );
            GA_ROHandleF  rohCur = fluid_gdp->findFloatTuple( GA_ATTRIB_POINT, "curvature", 1 );
            GA_ROHandleF  rohDen = fluid_gdp->findFloatTuple( GA_ATTRIB_POINT, "density", 1 );

            if( rohPos.isValid() )
            {
                pts.position.initialize( np, kHost );
                rohPos.getBlock( startoff, np, (UT_Vector3*)pts.position.pointer() );
            }

            if( rohAdj.isValid() )
            {
                pts.adjacency.initialize( np, kHost );
                rohAdj.getBlock( startoff, np, pts.adjacency.pointer() );
            }

            if( rohDen.isValid() )
            {
                pts.density.initialize( np, kHost );
                rohDen.getBlock( startoff, np, pts.density.pointer() );
            }

            if( rohVor.isValid() )
            {
                pts.vorticity.initialize( np, kHost );
                rohVor.getBlock( startoff, np, pts.vorticity.pointer() );
            }

            if( rohCur.isValid() )
            {
                pts.curvature.initialize( np, kHost );
                rohCur.getBlock (startoff, np, pts.curvature.pointer() );
            }

            DM_ParticleRender::Data data;

            data.pts = &pts;
            data.type = houEvalInt( this, "typemenu" );
            data.adjacencyUnder = houEvalFloat( this, "adja" );
            data.curvatureOver = houEvalFloat( this, "curv" );
            data.color = houEvalVec3f( this, "color" );

            DM_ParticleRender::map[size_t(this)] = data;
        }
    }

    return error();
}

