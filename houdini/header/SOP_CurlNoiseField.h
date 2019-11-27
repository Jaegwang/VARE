//----------------------//
// SOP_CurlNoiseField.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.11.30                               //
//-------------------------------------------------------//

#pragma once

#include <HouCommon.h>
#include <Bora.h>

class SOP_CurlNoiseField : public SOP_Node
{
    private:

        Noise _noise;

    public:
    
        SOP_CurlNoiseField( OP_Network* net, const char* name, OP_Operator* entry );
        virtual ~SOP_CurlNoiseField();

        static const char* uniqueName;
        static const char* labelName;
        static const char* menuName;
        static const char* inputLabels[];

        static PRM_Template nodeParameters[];

        static OP_Node *constructor( OP_Network *net, const char *name, OP_Operator *entry );        
        
    protected:

        virtual OP_ERROR cookMySop( OP_Context& context );
};

