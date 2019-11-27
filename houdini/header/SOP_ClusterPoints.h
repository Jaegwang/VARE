//---------------------//
// SOP_ClusterPoints.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.09.05                               //
//-------------------------------------------------------//

#pragma once

#include <HouCommon.h>
#include <Bora.h>

class SOP_ClusterPoints : public SOP_Node
{
    private:

        Vec3fArray _pointArray;
        Vec3fArray _centroidArray;
        IndexArray _IdArray, _secArray;

        HashGridTable _table;

        Vec3fArray _colorArray;

    public:
    
        SOP_ClusterPoints( OP_Network* net, const char* name, OP_Operator* entry );
        virtual ~SOP_ClusterPoints();

        static OP_Node *constructor( OP_Network *net, const char *name, OP_Operator *entry );

        static const char* uniqueName;
        static const char* labelName;
        static const char* menuName;
        static const char* inputLabels[];

        static PRM_Template nodeParameters[];
        
    protected:

        virtual OP_ERROR cookMySop( OP_Context& context );
};

