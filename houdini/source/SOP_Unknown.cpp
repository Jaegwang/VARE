//-----------------//
// SOP_Unknown.cpp //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.05.31                               //
//-------------------------------------------------------//

#include <SOP_Unknown.h>

const char* SOP_Unknown::uniqueName = "bora_unknown";
const char* SOP_Unknown::labelName = "Bora Unknown";
const char* SOP_Unknown::menuName = "Bora";

const char* SOP_Unknown::inputLabels[] = {
    "Points"
};

PRM_Template SOP_Unknown::nodeParameters[]=
{
    houIntParameter( "iter", "Iteration", 20 ),
    houFloatParameter( "clusterrad", "Cluster Radius", 0.5f ),
    houBoolParameter( "boolcheck", "Check", false ),
    PRM_Template()
};

OP_Node*
SOP_Unknown::constructor( OP_Network*net, const char *name, OP_Operator* entry )
{
    return new SOP_Unknown( net, name, entry );
}

SOP_Unknown::SOP_Unknown( OP_Network *net, const char *name, OP_Operator *entry )
: SOP_Node(net, name, entry)
{
}

SOP_Unknown::~SOP_Unknown()
{
}

OP_ERROR
SOP_Unknown::cookMySop( OP_Context &context )
{
    OP_AutoLockInputs inputs(this);
    if (inputs.lock(context) >= UT_ERROR_ABORT) return error();
    
    const GU_Detail *ptsgdp = inputGeo( 0 );

    if( ptsgdp )
    {
        _positions.initialize( ptsgdp->getNumPoints() );
        _velocities.initialize( ptsgdp->getNumPoints() );

        GA_ROHandleV3 rohPos = ptsgdp->findFloatTuple( GA_ATTRIB_POINT, "P", 3 );
        GA_ROHandleV3 rohVel = ptsgdp->findFloatTuple( GA_ATTRIB_POINT, "v", 3 );

        size_t n=0;
        GA_Offset ptoff;

        GA_FOR_ALL_PTOFF( ptsgdp, ptoff )
        {
            if( rohPos != 0 )
            {
                UT_Vector3 p = rohPos.get( ptoff );
                _positions[n] = Vec3f( p.x(), p.y(), p.z() );
            }

            if( rohVel != 0  )
            {
                UT_Vector3 v = rohVel.get( ptoff );
                _velocities[n] = Vec3f( v.x(), v.y(), v.z() );
            }

            n++;
        }
    }
    
    // geometry detail pointer;
    gdp->clearAndDestroy();

    // generate new child particles;
    const size_t N = _positions.size();

    for( size_t i=0; i<N; ++i )
    {
        _positions.append( _positions[i] + Vec3f( 0.f, 2.f, 0.f ) );
        _velocities.append( Vec3f( 0.f ) );
    }


    // output
    GA_RWHandleV3 velHnd = GA_RWHandleV3( gdp->addFloatTuple( GA_ATTRIB_POINT, "v", 3 ) );

    for( int n=0; n<_positions.size(); ++n )
    {
        const Vec3f& pos = _positions[n];
        const Vec3f& vel = _velocities[n];

        GA_Offset ptoff = gdp->appendPointOffset();

        gdp->setPos3( ptoff, UT_Vector3(pos.x, pos.y, pos.z) );

        velHnd.set( ptoff, UT_Vector3(vel.x, vel.y, vel.z) );
    }    

    gdp->bumpAllDataIds();

    inputs.unlock();
    return error();
}

