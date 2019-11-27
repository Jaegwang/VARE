//-----------------------//
// SOP_ClusterPoints.cpp //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.09.05                               //
//-------------------------------------------------------//

#include <SOP_ClusterPoints.h>

const char* SOP_ClusterPoints::uniqueName = "boraclusterpoints";
const char* SOP_ClusterPoints::labelName = "Bora Cluster points";
const char* SOP_ClusterPoints::menuName = "Bora";

const char* SOP_ClusterPoints::inputLabels[] = {
    "Points"
};

PRM_Template SOP_ClusterPoints::nodeParameters[]=
{
    houIntParameter( "iteration", "Iteration", 20 ),
    houFloatParameter( "clusterrad", "Cluster Radius", 0.5f ),
    PRM_Template()
};

OP_Node*
SOP_ClusterPoints::constructor( OP_Network*net, const char *name, OP_Operator* entry )
{
    return new SOP_ClusterPoints( net, name, entry );
}

SOP_ClusterPoints::SOP_ClusterPoints( OP_Network *net, const char *name, OP_Operator *entry )
: SOP_Node(net, name, entry)
{
}

SOP_ClusterPoints::~SOP_ClusterPoints()
{
    _pointArray.finalize();
    _centroidArray.finalize();
    _IdArray.finalize();
}

OP_ERROR
SOP_ClusterPoints::cookMySop( OP_Context &context )
{
    OP_AutoLockInputs inputs(this);
    if (inputs.lock(context) >= UT_ERROR_ABORT) return error();

    duplicateSource(0, context);

    gdp->addFloatTuple( GA_ATTRIB_POINT, "Cd", 3 );   
    gdp->addIntTuple( GA_ATTRIB_POINT, "cluster", 1 );
       
    GA_ROHandleV3 rohPos( gdp->findFloatTuple( GA_ATTRIB_POINT, "P" , 3 ) );
    GA_RWHandleV3 rwhCol( gdp->findFloatTuple( GA_ATTRIB_POINT, "Cd", 3 ) );    
    //GA_RWHandleI  rwhId ( gdp->findIntTuple( GA_ATTRIB_POINT, "cluster", 1 ) );

    const GA_Offset startoff = gdp->pointOffset( 0 );
    const size_t np = gdp->getNumPoints();

    _pointArray.initialize( np, kUnified );
    _colorArray.initialize( np, kUnified );

    rohPos.getBlock( startoff, np, (UT_Vector3*)_pointArray.pointer() );    

    KMeanClusterPoints( _centroidArray, _IdArray, _secArray, _table, _pointArray, houEvalFloat(this, "clusterrad"), houEvalInt(this,"iteration") );

    for( size_t n=0; n<_pointArray.size(); ++n )
    {
        const int& s = _IdArray[n];
        _colorArray[n] = Vec3f( Rand(s*35415), Rand(s*54223), Rand(s*83242) );
    }

    //rwhId.setBlock( startoff, np, _IdArray.pointer() );
    rwhCol.setBlock( startoff, np, (UT_Vector3*)_colorArray.pointer() );

    //rwhId->bumpDataId();
    rwhCol->bumpDataId();
    return error();
}

