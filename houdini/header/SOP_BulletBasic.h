//-------------------//
// SOP_BulletBasic.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.09.11                               //
//-------------------------------------------------------//

#pragma once

#include <HouCommon.h>
#include <Bora.h>

class SOP_BulletBasic : public SOP_Node
{
    private:

        PointArray _positions;
        Array< UT_Matrix3F > _transforms;        

        /* BulletPysics */
        BulletTest _bullet_test;

    private:

        int _cookedFrame=-1;

    public:
    
        SOP_BulletBasic( OP_Network* net, const char* name, OP_Operator* entry );
        virtual ~SOP_BulletBasic();

        static const char* uniqueName;
        static const char* labelName;
        static const char* menuName;
        static const char* inputLabels[];

        static PRM_Template nodeParameters[];

        static OP_Node *constructor( OP_Network *net, const char *name, OP_Operator *entry );        
        
    protected:

        virtual OP_ERROR cookMySop( OP_Context& context );

    private:

        void initialize( OP_Context& context );
        void simulate( OP_Context& context );
};

