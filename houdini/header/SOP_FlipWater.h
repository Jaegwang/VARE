//-----------------//
// SOP_FlipWater.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2019.02.21                               //
//-------------------------------------------------------//

#pragma once
#include <HouCommon.h>

class SOP_FlipWater : public SOP_Node
{
    private:

        Particles _pts;

        FLIPMethod _flip;

        Grid _grid;
        
        int cookedFrame=-1;

    public:

        SOP_FlipWater( OP_Network* net, const char* name, OP_Operator* entry );
        virtual ~SOP_FlipWater();

        static const char* uniqueName;
        static const char* labelName;
        static const char* menuName;
        static const char* inputLabels[];

        static PRM_Template nodeParameters[];

        static OP_Node* constructor( OP_Network* net, const char* name, OP_Operator* entry );

    public:

        void initialize( OP_Context& context );
        void idle( OP_Context& context );
        
    protected:

        virtual OP_ERROR cookMySop( OP_Context& context );
};

