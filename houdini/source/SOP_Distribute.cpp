//--------------------//
// SOP_Distribute.cpp //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.06.22                               //
//-------------------------------------------------------//

#include <SOP_Distribute.h>

const char* SOP_Distribute::uniqueName = "bora_distribute";
const char* SOP_Distribute::labelName = "Bora Distribute";
const char* SOP_Distribute::menuName = "Bora";

const char* SOP_Distribute::inputLabels[] = {
    "Points"
};

PRM_Template SOP_Distribute::nodeParameters[]=
{
    houIntParameter( "iter", "Iteration", 20 ),
    houFloatParameter( "clusterrad", "Cluster Radius", 0.5f ),
    houBoolParameter( "boolcheck", "Check", false ),
    PRM_Template()
};

OP_Node*
SOP_Distribute::constructor( OP_Network*net, const char *name, OP_Operator* entry )
{
    return new SOP_Distribute( net, name, entry );
}

SOP_Distribute::SOP_Distribute( OP_Network *net, const char *name, OP_Operator *entry )
: SOP_Node(net, name, entry)
{
}

SOP_Distribute::~SOP_Distribute()
{
}

OP_ERROR
SOP_Distribute::cookMySop( OP_Context &context )
{
    OP_AutoLockInputs inputs(this);
    if (inputs.lock(context) >= UT_ERROR_ABORT) return error();
    
    const GU_Detail *ptsgdp = inputGeo( 0 );

    _positions.initialize( 0 );
    _normals.initialize( 0 );
    _tangents.initialize( 0 );
    _binormals.initialize( 0 );
    _transforms.initialize( 0 );

    _positions.reserve( 10000 );
    _normals.reserve( 10000 ); 
    _tangents.reserve( 10000 );
    _binormals.reserve( 10000 );
    _transforms.reserve( 10000 );    

    if( ptsgdp )
    {
        
        const GA_Offset primNum = ptsgdp->getNumPrimitiveOffsets();
        const GA_PrimitiveList& primList = ptsgdp->getPrimitiveList();

        for( GA_Offset p=0; p<primNum; ++p )
        {
            const GA_Primitive* prim = primList.get( p );

            UT_Vector3 _p0 = prim->getPos3( 0 );
            UT_Vector3 _p1 = prim->getPos3( 1 );
            UT_Vector3 _p2 = prim->getPos3( 2 );

            Vec3f p0( _p0.x(), _p0.y(), _p0.z() );
            Vec3f p1( _p1.x(), _p1.y(), _p1.z() );
            Vec3f p2( _p2.x(), _p2.y(), _p2.z() );

            const float w0 = Rand( p*534345, 0.f, 1.f    );
            const float w1 = Rand( p*322366, 0.f, 1.f-w0 );
            const float w2 = 1.f-w0-w1;

            const float t0 = Rand( p*654645, 0.f, 1.f    );
            const float t1 = Rand( p*454366, 0.f, 1.f-t0 );
            const float t2 = 1.f-t0-t1;
            
            Vec3f pos = p0*w0 + p1*w1 + p2*w2;
            Vec3f u = p0*t0 + p1*t1 + p2*t2;

            Vec3f tan = ( u-pos ).normalized();
            Vec3f nor = ( (p2-p0)^(p1-p0) ).normalized();
            Vec3f bin = ( nor^tan ).normalized();
            
            _positions.append( pos );

            _normals.append( nor );
            _tangents.append( tan );
            _binormals.append( bin );

            UT_Matrix3F mat;
            mat[0] = UT_Vector3( bin.x, bin.y, bin.z );
            mat[1] = UT_Vector3( nor.x, nor.y, nor.z );
            mat[2] = UT_Vector3( tan.x, tan.y, tan.z );

            _transforms.append( mat );
        }
    }
  
    // generate new child particles;
    const size_t N = _positions.size();

    // geometry detail pointer;
    gdp->clearAndDestroy();
    gdp->appendPointBlock( N );

    // output
    GA_RWHandleV3 posHnd = GA_RWHandleV3( gdp->addFloatTuple( GA_ATTRIB_POINT, "P", 3 ) );
    GA_RWHandleV3 norHnd = GA_RWHandleV3( gdp->addFloatTuple( GA_ATTRIB_POINT, "nor", 3 ) );
    GA_RWHandleV3 tanHnd = GA_RWHandleV3( gdp->addFloatTuple( GA_ATTRIB_POINT, "tan", 3 ) );
    GA_RWHandleV3 binHnd = GA_RWHandleV3( gdp->addFloatTuple( GA_ATTRIB_POINT, "bin", 3 ) );

    GA_RWHandleM3 trmHnd = GA_RWHandleM3( gdp->addFloatTuple( GA_ATTRIB_POINT, "transform", 9 ) );
    
    posHnd.setBlock( 0, _positions.size(), (UT_Vector3*)_positions.pointer() );
    norHnd.setBlock( 0, _normals.size(), (UT_Vector3*)_normals.pointer() );
    tanHnd.setBlock( 0, _tangents.size(), (UT_Vector3*)_tangents.pointer() );
    binHnd.setBlock( 0, _binormals.size(), (UT_Vector3*)_binormals.pointer() );
    trmHnd.setBlock( 0, _transforms.size(), (UT_Matrix3F*)_transforms.pointer() );

    inputs.unlock();
    return error();
}

