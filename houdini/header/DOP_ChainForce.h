//------------------//
// DOP_ChainForce.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.09.06                               //
//-------------------------------------------------------//

#pragma once

#include <HouCommon.h>
#include <Bora.h>

class DOP_ChainForce : public DOP_Node
{
	private:

		Vec3fArray _pointArray, _veloArray;

		ChainForceMethod _chainForce;

	public:

		DOP_ChainForce( OP_Network *net, const char *name, OP_Operator *op );
		
		virtual ~DOP_ChainForce();

		static const char* uniqueName;
		static const char* labelName;
		static const char* menuName;

    	static PRM_Template nodeParameters[];

		static OP_Node *constructor(OP_Network *net, const char *name, OP_Operator *op );		

	protected:

	    virtual void processObjectsSubclass( fpreal time, int foroutputidx, const SIM_ObjectArray &objects, DOP_Engine &engine );

    	virtual void getInputInfoSubclass( int inputidx, DOP_InOutInfo &info ) const;
    	virtual void getOutputInfoSubclass( int inputidx, DOP_InOutInfo &info ) const;

};

