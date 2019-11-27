//----------------//
// BoraNodeData.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.12.06                               //
//-------------------------------------------------------//

#pragma once

#include <Bora.h>
#include <MayaCommon.h>

class BoraNodeData : public MPxData
{
    public:

        static MTypeId id;
        static MString typeName;

    public:

        MTypeId nodeId;
        void* nodePtr=0;

    public:

        BoraNodeData();

        static void* creator();

        virtual MTypeId typeId() const;
        virtual MString name() const;
        virtual void copy( const MPxData& other );
};

