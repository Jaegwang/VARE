//---------------//
// SOP_Unknown.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.05.31                               //
//-------------------------------------------------------//

#pragma once

#include <HouCommon.h>
#include <Bora.h>

class SOP_Unknown : public SOP_Node
{
    private:

        PointArray _positions;
        Vec3fArray _velocities;

    public:
    
        SOP_Unknown( OP_Network* net, const char* name, OP_Operator* entry );
        virtual ~SOP_Unknown();

        static OP_Node *constructor( OP_Network *net, const char *name, OP_Operator *entry );

        static const char* uniqueName;
        static const char* labelName;
        static const char* menuName;
        static const char* inputLabels[];

        static PRM_Template nodeParameters[];
        
    protected:

        virtual OP_ERROR cookMySop( OP_Context& context ); // maya compute();
};

