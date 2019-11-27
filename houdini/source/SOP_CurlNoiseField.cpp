//------------------------//
// SOP_CurlNoiseField.cpp //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.11.30                               //
//-------------------------------------------------------//

#include <SOP_CurlNoiseField.h>

const char* SOP_CurlNoiseField::uniqueName = "boracurlnoisefield";
const char* SOP_CurlNoiseField::labelName = "Bora Curl Noise Field";
const char* SOP_CurlNoiseField::menuName = "Bora";

const char* SOP_CurlNoiseField::inputLabels[] =
{
    "Box"
};

PRM_Template SOP_CurlNoiseField::nodeParameters[] =
{
    houFloatParameter( "frequency" , "Frequency", 0.1f ),
    houFloatParameter( "offset"    , "Offset", 0.f ),
    houFloatParameter( "scale"     , "Scale", 1.f ),  
    houVec3fParameter( "addforce"  , "Add Force", Vec3f(0.f) ),
    PRM_Template()
};

OP_Node*
SOP_CurlNoiseField::constructor( OP_Network*net, const char *name, OP_Operator* entry )
{
    return new SOP_CurlNoiseField( net, name, entry );
}

SOP_CurlNoiseField::SOP_CurlNoiseField( OP_Network*net, const char* name, OP_Operator* entry )
: SOP_Node(net, name, entry)
{
}

SOP_CurlNoiseField::~SOP_CurlNoiseField()
{
}

OP_ERROR
SOP_CurlNoiseField::cookMySop( OP_Context &context )
{
    OP_AutoLockInputs inputs( this );
    if (inputs.lock(context) >= UT_ERROR_ABORT) return error();

    const GU_Detail *boxgdp = inputGeo( 0 );
    
    GA_ROHandleV3 rohSize = boxgdp->findFloatTuple( GA_ATTRIB_DETAIL, "gridsize", 3 );
    GA_ROHandleV3 rohCenter = boxgdp->findFloatTuple( GA_ATTRIB_DETAIL, "gridcenter", 3 );
    GA_ROHandleF rohVoxel = boxgdp->findFloatTuple( GA_ATTRIB_DETAIL, "voxelsize", 1 );

    if( rohVoxel.isValid() == false ) return error();

    UT_Vector3 gSize = rohSize.get( GA_Offset(0) );
    UT_Vector3 gCenter = rohCenter.get( GA_Offset(0) );

    OP_Node:flags().timeDep = 1;

    const float dt = 1.f/24.f;

    Vec3f gridcenter = Vec3f( gCenter.x(), gCenter.y(), gCenter.z() );
    Vec3f gridsize = Vec3f( gSize.x(), gSize.y(), gSize.z() );
    Vec3f gridMin = gridcenter-(gridsize*0.5f);
    Vec3f gridMax = gridcenter+(gridsize*0.5f);

    const float h = rohVoxel.get( GA_Offset(0) ) * 8.f;

    CH_Manager *chman = OPgetDirector()->getChannelManager();
    int currframe = chman->getSample(context.getTime());    

    Grid grid;
    grid.initialize( h, gridMin, gridMax );

    duplicateSource( 0, context );

    Vec3f addForce = houEvalVec3f( this, "addforce" );    

    _noise.frequency = Vec4f( houEvalFloat( this, "frequency" ) );
    _noise.offset = Vec4f( houEvalFloat( this, "offset" ) );
    _noise.scale = houEvalFloat( this, "scale" );
    _noise.add = addForce;

    for( int i=0; i<grid.nx(); ++i )
    for( int j=0; j<grid.ny(); ++j )
    {
        int k = grid.nz()/2;
        Vec3f vpos = grid.cellCenter( i,j,k );

        const int num = 3*_noise.scale;
        GEO_PrimPoly* poly = GEO_PrimPoly::build( gdp, num, true );

        for( int x=0; x<num; ++x )
        {
            GA_Offset ptoff = poly->getPointOffset( x );

            const Vec3f wpos = grid.worldPoint( vpos );
            gdp->setPos3( ptoff, UT_Vector3( wpos.x, wpos.y, wpos.z ) );

            vpos += _noise.curl( vpos.x,vpos.y,vpos.z, currframe*dt, 1 ) * (1.f/(float)num);
        }
    }

    {
        GA_RWHandleF  frequencyHnd = GA_RWHandleF( gdp->addFloatTuple( GA_ATTRIB_DETAIL, "curl_frequency", 1 ) );
        GA_RWHandleF  offsetHnd = GA_RWHandleF( gdp->addFloatTuple( GA_ATTRIB_DETAIL, "curl_offset", 1 ) );
        GA_RWHandleF  scaleHnd = GA_RWHandleF( gdp->addFloatTuple( GA_ATTRIB_DETAIL, "curl_scale", 1 ) );
        GA_RWHandleV3 addforceHdn = GA_RWHandleV3( gdp->addFloatTuple( GA_ATTRIB_DETAIL, "curl_add", 3 ) );

        frequencyHnd.set( 0, houEvalFloat( this, "frequency" ) );
        offsetHnd.set( 0, houEvalFloat( this, "offset" ) );
        scaleHnd.set( 0, houEvalFloat( this, "scale" ) );
        addforceHdn.set( 0, UT_Vector3( addForce.x, addForce.y, addForce.z ) );
    }

    return error();
}

