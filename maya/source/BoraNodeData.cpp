//------------------//
// BoraNodeData.cpp //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.12.06                               //
//-------------------------------------------------------//

#include <BoraNodeData.h>

MTypeId BoraNodeData::id( 0x200001 );
MString BoraNodeData::typeName( "BoraNodeData" );

BoraNodeData::BoraNodeData() : MPxData()
{
}

void*
BoraNodeData::creator()
{
	return new BoraNodeData();
}

MTypeId
BoraNodeData::typeId() const
{
	return id;
}

MString
BoraNodeData::name() const
{
	return typeName;
}

void
BoraNodeData::copy( const MPxData& other )
{
	const BoraNodeData& data = (const BoraNodeData&)other;

    nodeId = data.nodeId;
	nodePtr = data.nodePtr;
}

