//--------------------//
// SOP_DiffuseWater.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.09.05                               //
//-------------------------------------------------------//

#pragma once
#include <HouCommon.h>

class SOP_DiffuseWater : public SOP_Node
{
    private:

        DiffuseWaterMethod _diffWater;
        Particles _fluidpts;

        Vec3f _gridSize;
        Vec3f _gridCenter;
        float _voxelSize=0.f;

        int cookedFrame = -1;

    public:
    
        SOP_DiffuseWater( OP_Network* net, const char* name, OP_Operator* entry );
        virtual ~SOP_DiffuseWater();

        static OP_Node* constructor( OP_Network *net, const char *name, OP_Operator *entry );

        static const char* uniqueName;
        static const char* labelName;
        static const char* menuName;
        static const char* inputLabels[];

        static PRM_Template nodeParameters[];

        void initialize( OP_Context& context );
        void idle( OP_Context& context );

    protected:

        virtual OP_ERROR cookMySop( OP_Context& context );

        virtual bool updateParmsFlags();
};

