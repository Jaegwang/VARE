//-----------------//
// DOP_Unknown.cpp //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.06.07                               //
//-------------------------------------------------------//

#include <DOP_Unknown.h>

const char* DOP_Unknown::uniqueName = "bora_doptest";
const char* DOP_Unknown::labelName = "Bora DOPTest";

PRM_Template DOP_Unknown::myTemplateList[] = 
{
    PRM_Template()
};

OP_Node*
DOP_Unknown::myConstructor( OP_Network *net, const char *name, OP_Operator *op )
{
    return new DOP_Unknown(net, name, op);
}

DOP_Unknown::DOP_Unknown( OP_Network *net, const char *name, OP_Operator *op )
: DOP_Node(net, name, op)
{
}

DOP_Unknown::~DOP_Unknown()
{
}

void
DOP_Unknown::processObjectsSubclass( fpreal time, int foroutputidx, const SIM_ObjectArray &objects, DOP_Engine &engine )
{
	//for( int i=0; i<objects.entries(); ++i )
	if( objects.entries() > 0 )	
	{
		//SIM_Object* obj = objects.findObjectById( i );
		SIM_Object* obj = objects.findObjectById( 0 );

		SIM_GeometryCopy* geo(0);
		geo = SIM_DATA_GET( *obj, SIM_GEOMETRY_DATANAME, SIM_GeometryCopy );

		if( geo )
		{
			SIM_GeometryAutoWriteLock lock(geo);
			GU_Detail* gdp = &lock.getGdp();

		    GA_RWHandleV3 rwhPos;
    		GA_ROHandleV3 rohPos;

    		rwhPos = GA_RWHandleV3( gdp->findFloatTuple( GA_ATTRIB_POINT, "P", 3 ) );
    		rohPos = GA_ROHandleV3( gdp->findFloatTuple( GA_ATTRIB_POINT, "P", 3 ) );
			
			UT_Vector3 add( 0.1f, 0.f, 0.f );
			
			GA_Offset ptoff;
			GA_FOR_ALL_PTOFF( gdp, ptoff )
			{
				rwhPos.set( ptoff, rohPos.get(ptoff) + add );
			}

			gdp->findFloatTuple( GA_ATTRIB_POINT, "P", 3 )->bumpDataId();
			
		}
	}	
}

void
DOP_Unknown::getInputInfoSubclass( int inputidx, DOP_InOutInfo &info ) const
{
    // Our first input is an object input.
    // Our remaining inputs are data inputs.
    if( inputidx == 0 )
		info = DOP_InOutInfo(DOP_INOUT_OBJECTS, false);
    else
		info = DOP_InOutInfo(DOP_INOUT_DATA, true);
}

void
DOP_Unknown::getOutputInfoSubclass( int outputidx, DOP_InOutInfo &info ) const
{
    // Our single output is an object output.
    info = DOP_InOutInfo(DOP_INOUT_OBJECTS, false);
}

