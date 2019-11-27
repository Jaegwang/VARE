//--------------------//
// SOP_ParticleView.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2019.04.16                               //
//-------------------------------------------------------//

#pragma once
#include <HouCommon.h>

class SOP_ParticleView : public SOP_Node
{
    private:

        Particles pts;

    public:
    
        SOP_ParticleView( OP_Network* net, const char* name, OP_Operator* entry );
        virtual ~SOP_ParticleView();

        static OP_Node* constructor( OP_Network *net, const char *name, OP_Operator *entry );

        static const char* uniqueName;
        static const char* labelName;
        static const char* menuName;
        static const char* inputLabels[];

        static PRM_Template nodeParameters[];
        
    protected:

        virtual OP_ERROR cookMySop( OP_Context& context );

        virtual bool updateParmsFlags();
};

