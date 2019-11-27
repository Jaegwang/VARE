//---------------//
// DOP_Unknown.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.06.07                               //
//-------------------------------------------------------//

#pragma once
#include <HouCommon.h>

class DOP_Unknown : public DOP_Node
{
	public:

		DOP_Unknown( OP_Network *net, const char *name, OP_Operator *op );
		
		virtual ~DOP_Unknown();
		
		static OP_Node *myConstructor(OP_Network *net, const char *name, OP_Operator *op );

		static const char* uniqueName;
		static const char* labelName;

    	static PRM_Template myTemplateList[];

	protected:

	    virtual void processObjectsSubclass( fpreal time, int foroutputidx, const SIM_ObjectArray &objects, DOP_Engine &engine );

    	virtual void getInputInfoSubclass( int inputidx, DOP_InOutInfo &info) const;
    	virtual void getOutputInfoSubclass(int inputidx, DOP_InOutInfo &info) const;

};

