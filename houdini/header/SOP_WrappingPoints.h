//----------------------//
// SOP_WrappingPoints.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.09.10                               //
//-------------------------------------------------------//

#pragma once
#include <HouCommon.h>

class SOP_WrappingPoints : public SOP_Node
{
    private:

        Vec3fArray _uvwArray;

        Image _image;

    public:

        SOP_WrappingPoints( OP_Network* net, const char* name, OP_Operator* entry );
        virtual ~SOP_WrappingPoints();

        static const char* uniqueName;
        static const char* labelName;
        static const char* menuName;
        static const char* inputLabels[];

        static PRM_Template nodeParameters[];

        static OP_Node *constructor( OP_Network *net, const char *name, OP_Operator *entry );
        
    protected:

        virtual OP_ERROR cookMySop( OP_Context& context );

};

