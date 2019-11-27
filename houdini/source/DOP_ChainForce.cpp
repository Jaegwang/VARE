//--------------------//
// DOP_ChainForce.cpp //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.09.05                               //
//-------------------------------------------------------//

#include <DOP_ChainForce.h>

const char* DOP_ChainForce::uniqueName = "bora_chainforce";
const char* DOP_ChainForce::labelName = "Bora Chain Force";
const char* DOP_ChainForce::menuName = "Bora";

PRM_Template
DOP_ChainForce::nodeParameters[]=
{
    houIntParameter( "iter", "Iteration", 20 ),
    houFloatParameter( "clusterrad" , "Cluster Radius", 2.f ),
	houFloatParameter( "forcescale" , "Force Scale", 0.1f ),
    PRM_Template()
};

OP_Node*
DOP_ChainForce::constructor( OP_Network *net, const char *name, OP_Operator *op )
{
    return new DOP_ChainForce(net, name, op);
}

DOP_ChainForce::DOP_ChainForce( OP_Network *net, const char *name, OP_Operator *op )
: DOP_Node(net, name, op)
{
}

DOP_ChainForce::~DOP_ChainForce()
{
}

void
DOP_ChainForce::processObjectsSubclass( fpreal time, int foroutputidx, const SIM_ObjectArray &objects, DOP_Engine &engine )
{
	const float dt = 1.f/24.f;

	const int simNum = engine.getNumSimulationObjects();
	std::vector<SIM_Object*> simObjects;

	for( int i=0; i<simNum; ++i )
	{
		SIM_Object* obj = objects.findObjectById( i );
		if( obj ) simObjects.push_back( obj );
	}
	
	if( !simObjects.empty() )
	{
		SIM_Object* obj = simObjects[0];
		if( obj )
		{
			SIM_GeometryCopy* geo(0);
			geo = SIM_DATA_GET( *obj, SIM_GEOMETRY_DATANAME, SIM_GeometryCopy );

			if( geo )
			{
				SIM_GeometryAutoWriteLock lock(geo);
				GU_Detail* gdp = &lock.getGdp();

				if( gdp )
				{
					GA_ROHandleV3 rohPos = GA_ROHandleV3( gdp->findFloatTuple( GA_ATTRIB_POINT, "P", 3 ) );
					GA_RWHandleV3 rwhVel = GA_RWHandleV3( gdp->findFloatTuple( GA_ATTRIB_POINT, "v", 3 ) );

					const size_t N = gdp->getNumPoints();

					if( N )
					{	
						_pointArray.initialize( N, MemorySpace::kUnified );
						_veloArray.initialize( N, MemorySpace::kUnified );

						rohPos.getBlock( 0, N, (UT_Vector3*)_pointArray.pointer() );
						rwhVel.getBlock( 0, N, (UT_Vector3*)_veloArray.pointer() );
						const float radius = houEvalFloat( this, "clusterrad" );
						const float fScale = houEvalFloat( this, "forcescale" );
						const int iter = houEvalInt( this, "iter" );

						_chainForce.force( _pointArray, _veloArray, fScale, radius, iter );

						GA_Offset startoff = gdp->pointOffset( 0 );

						rwhVel.setBlock( startoff, N, (UT_Vector3*)_veloArray.pointer() );
						rwhVel->bumpDataId();						
					}

				}
			}
		}
	}
}

void
DOP_ChainForce::getInputInfoSubclass( int inputidx, DOP_InOutInfo &info ) const
{
    // Our first input is an object input.
    // Our remaining inputs are data inputs.
    if( inputidx == 0 )
		info = DOP_InOutInfo(DOP_INOUT_OBJECTS, false);
    else
		info = DOP_InOutInfo(DOP_INOUT_DATA, true);
}

void
DOP_ChainForce::getOutputInfoSubclass( int outputidx, DOP_InOutInfo &info ) const
{
    // Our single output is an object output.
    info = DOP_InOutInfo(DOP_INOUT_OBJECTS, false);
}

