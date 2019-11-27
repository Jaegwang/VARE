//------------//
// BoraData.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2017.09.18                               //
//-------------------------------------------------------//

#ifndef _BoraData_h_
#define _BoraData_h_

#include <MayaCommon.h>

class BoraData : public MPxData
{
    public:
        
        static MTypeId id;
        static MString typeName;

        void* nodePtr=0;
        MString nodeName;
        MTypeId nodeId;        

    public:

        BoraData();

        static void* creator();

        virtual MTypeId typeId() const;

        virtual MString name() const;

        virtual void copy( const MPxData& other );
};

#endif

