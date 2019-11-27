//--------------//
// BoraData.cpp //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2017.09.18                               //
//-------------------------------------------------------//

#include <BoraData.h>

MTypeId BoraData::id( 0x200001 );
MString BoraData::typeName( "BoraData" );

BoraData::BoraData()
: MPxData()
{
}

void*
BoraData::creator()
{
	return new BoraData();
}

MTypeId
BoraData::typeId() const
{
	return id;
}

MString
BoraData::name() const
{
	return typeName;
}

void
BoraData::copy( const MPxData& other )
{
    const BoraData& data = (const BoraData&)other;
    nodePtr = data.nodePtr;
    nodeName = data.nodeName;
    nodeId = data.nodeId;
}

