//------------------//
// SOP_Distribute.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.06.22                               //
//-------------------------------------------------------//

#pragma once

#include <HouCommon.h>
#include <Bora.h>

class SOP_Distribute : public SOP_Node
{
    private:

        PointArray _positions;
        Vec3fArray _normals, _tangents, _binormals;

        Array< UT_Matrix3F > _transforms;

    public:
    
        SOP_Distribute( OP_Network* net, const char* name, OP_Operator* entry );
        virtual ~SOP_Distribute();

        static OP_Node *constructor( OP_Network *net, const char *name, OP_Operator *entry );

        static const char* uniqueName;
        static const char* labelName;
        static const char* menuName;
        static const char* inputLabels[];

        static PRM_Template nodeParameters[];
        
    protected:

        virtual OP_ERROR cookMySop( OP_Context& context ); // maya compute();
};

