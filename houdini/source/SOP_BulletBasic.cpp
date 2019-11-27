//---------------------//
// SOP_BulletBasic.cpp //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.09.11                               //
//-------------------------------------------------------//

#include <SOP_BulletBasic.h>

const char* SOP_BulletBasic::uniqueName = "borabulletbasic";
const char* SOP_BulletBasic::labelName = "Bora Bullet Basic";
const char* SOP_BulletBasic::menuName = "Bora";

const char* SOP_BulletBasic::inputLabels[] =
{
    "Source Points"
};

PRM_Template SOP_BulletBasic::nodeParameters[] =
{
    houIntParameter( "startframe" , "Start Frame", 1 ),
    PRM_Template()
};

OP_Node*
SOP_BulletBasic::constructor( OP_Network*net, const char *name, OP_Operator* entry )
{
    return new SOP_BulletBasic( net, name, entry );
}

SOP_BulletBasic::SOP_BulletBasic( OP_Network*net, const char* name, OP_Operator* entry )
: SOP_Node(net, name, entry)
{


}

SOP_BulletBasic::~SOP_BulletBasic()
{
}

OP_ERROR
SOP_BulletBasic::cookMySop( OP_Context &context )
{
    OP_AutoLockInputs inputs( this );
    if (inputs.lock(context) >= UT_ERROR_ABORT) return error();


    OP_Node:flags().timeDep = 1;

    CH_Manager *chman = OPgetDirector()->getChannelManager();
    int currframe = chman->getSample(context.getTime());

    const int startframe = houEvalInt( this, "startframe" );


    if( currframe == startframe )
    {        
        initialize( context );
        _cookedFrame = currframe;
    }
    else if( currframe == _cookedFrame+1 )
    {
        simulate( context );
        _cookedFrame = currframe;
    }

    const int N = _bullet_test.origins.size();
    _positions.resize( N );
    _transforms.resize( N );

    for( int n=0; n<N; ++n )
    {
        const btVector3& bPos = _bullet_test.origins[n];
        const btMatrix3x3& bMat = _bullet_test.basises[n];

        _positions[n] = Vec3f( bPos[0], bPos[1], bPos[2] );
        _transforms[n][0] = UT_Vector3( bMat[0][0], bMat[1][0], bMat[2][0] );
        _transforms[n][1] = UT_Vector3( bMat[0][1], bMat[1][1], bMat[2][1] );
        _transforms[n][2] = UT_Vector3( bMat[0][2], bMat[1][2], bMat[2][2] );
    }

    gdp->stashAll();

    GA_Offset startoff = gdp->appendPointBlock( N );

    GA_RWHandleV3 posHnd = GA_RWHandleV3( gdp->addFloatTuple( GA_ATTRIB_POINT, "P", 3 ) );
    GA_RWHandleM3 trmHnd = GA_RWHandleM3( gdp->addFloatTuple( GA_ATTRIB_POINT, "transform", 9 ) );    

    posHnd.setBlock( startoff, N, (UT_Vector3*)_positions.pointer() );
    trmHnd.setBlock( startoff, N, (UT_Matrix3F*)_transforms.pointer() );


    std::stringstream ss; ss << (size_t)this;
    GA_RWHandleS rwh_id( gdp->addStringTuple( GA_ATTRIB_DETAIL, "bora_uid", 1 ) );
    rwh_id.set( GA_Offset(0), ss.str() );
    

    gdp->destroyStashed();

    return error();
}

void
SOP_BulletBasic::initialize( OP_Context& context )
{
    _bullet_test.initialize();

    CH_Manager *chman = OPgetDirector()->getChannelManager();
    int currframe = chman->getSample(context.getTime());   

    const GU_Detail* ptsgdp = inputGeo( 0 );
    if( ptsgdp ) 
    {
        const int num = ptsgdp->getNumPoints();
        GA_Offset startoff = ptsgdp->pointOffset( GA_Index(0) );

        GA_ROHandleV3 posHnd = GA_ROHandleV3( ptsgdp->findFloatTuple( GA_ATTRIB_POINT, "P", 3 ) );
        //GA_ROHandleV3 velHnd = GA_ROHandleV3( ptsgdp->findFloatTuple( GA_ATTRIB_POINT, "v", 3 ) );

        Vec3fArray posArr; posArr.initialize( num );
        //Vec3fArray velArr; velArr.initialize( num );

        posHnd.getBlock( startoff, num, (UT_Vector3*)posArr.pointer() );
        //velHnd.getBlock( startoff, num, (UT_Vector3*)velArr.pointer() );

        for( int n=0; n<num; ++n )
        {
            _bullet_test.addRigidBody( posArr[n], Vec3f(0.f) );
        }
    }        
}

void
SOP_BulletBasic::simulate(  OP_Context& context )
{
    _bullet_test.simulate();
}

