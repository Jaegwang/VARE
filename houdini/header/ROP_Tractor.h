//---------------//
// ROP_Tractor.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2019.04.10                               //
//-------------------------------------------------------//

#pragma once
#include <HouCommon.h>

class ROP_Tractor : public ROP_Node
{
    protected:

        ROP_Tractor( OP_Network* net, const char* name, OP_Operator* entry );
        virtual ~ROP_Tractor();

        virtual int startRender( int nframes, fpreal s, fpreal e );

        virtual ROP_RENDER_CODE renderFrame( fpreal time, UT_Interrupt* boss );

        virtual ROP_RENDER_CODE endRender();

    public:

        static const char* uniqueName;
        static const char* labelName;
        static const char* menuName;

        static PRM_Template nodeParameters[];

        static OP_Node* constructor( OP_Network* net, const char* name, OP_Operator* entry );

        static int submitCallback( void* data, int index, float time, const PRM_Template* tplate );
};

