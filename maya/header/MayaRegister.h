//----------------//
// MayaRegister.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.12.06                               //
//-------------------------------------------------------//

#pragma once
#include <MayaCommon.h>

// Data
#define RegisterData( plugin, class )\
{\
	MStatus s = MS::kSuccess;\
	s = plugin.registerData( class::typeName, class::id, class::creator );\
	if( !s ) { s.perror( MString("Failed to register: ") + class::typeName ); return s; }\
}

#define RegisterGeomData( plugin, class )\
{\
	MStatus s = MS::kSuccess;\
	s = plugin.registerData( class::typeName, class::id, class::creator, MPxData::kGeometryData );\
	if( !s ) { s.perror( MString("Failed to register: ") + class::typeName ); return s; }\
}

#define DeregisterData( plugin, class )\
{\
	MStatus s = MS::kSuccess;\
	s = plugin.deregisterData( class::id );\
	if( !s ) { s.perror( MString("Failed to deregister: ") + class::typeName ); return s; }\
}

// Node
#define RegisterNode( plugin, node )\
{\
	MStatus s = MS::kSuccess;\
    s = plugin.registerNode( node::name, node::id, node::creator, node::initialize, MPxNode::kDependNode );\
    if( !s ) { s.perror( MString("Failed to register: ") + node::name ); return s; }\
}

#define DeregisterNode( plugin, node )\
{\
	s = plugin.deregisterNode( node::id );\
	if( !s ) { s.perror( MString("Failed to deregister: ") + node::name ); return s; }\
}

// Locator Node
#define RegisterLocatorWithDrawOverride( plugin, node, drawOverride )\
{\
	MStatus s = MS::kSuccess;\
    s = plugin.registerNode( node::name, node::id, node::creator, node::initialize, MPxNode::kLocatorNode, &node::drawDbClassification );\
    if( !s ) { s.perror( MString("Failed to register: ") + node::name ); return s; }\
    s = MHWRender::MDrawRegistry::registerDrawOverrideCreator( node::drawDbClassification, node::drawRegistrantId, drawOverride::Creator );\
    if( !s ) { s.perror( MString("Failed to register: ") + node::drawRegistrantId ); return s; }\
}

#define DeregisterLocatorWithDrawOverride( plugin, node )\
{\
	s = MHWRender::MDrawRegistry::deregisterDrawOverrideCreator( node::drawDbClassification, node::drawRegistrantId );\
	if( !s ) { s.perror( MString("Failed to deregister: ") + node::name ); return s; }\
	s = plugin.deregisterNode( node::id );\
	if( !s ) { s.perror( MString("Failed to deregister: ") + node::name ); return s; }\
}

// Command
#define RegisterCommand( plugin, class )\
{\
	MStatus s = MS::kSuccess;\
	s = plugin.registerCommand( class::name, class::creator, class::newSyntax );\
	if( !s ) { s.perror( MString("Failed to register: ") + class::name ); return s; }\
}

#define DeregisterCommand( plugin, class )\
{\
	MStatus s;\
	s = plugin.deregisterCommand( class::name );\
	if( !s ) { s.perror( MString("Failed to deregister: ") + class::name ); return s; }\
}

// Shade Node
#define RegisterShapeWithDrawOverride( plugin, node, drawOverride )\
{\
	MStatus s = MS::kSuccess;\
	s = plugin.registerShape( node::name, node::id, node::creator, node::initialize, NULL, &node::drawDbClassification );\
	if( !s ) { s.perror( MString("Failed to register: ") + node::name ); return s; }\
	s = MHWRender::MDrawRegistry::registerDrawOverrideCreator( node::drawDbClassification, node::drawRegistrantId, drawOverride::Creator );\
	if( !s ) { s.perror( MString("Failed to register: ") + node::drawRegistrantId ); return s; }\
}

#define DeregisterShapeWithDrawOverride( plugin, node )\
{\
	MStatus s = MS::kSuccess;\
	s = plugin.deregisterNode( node::id );\
	if( !s ) { s.perror( MString("Failed to deregister: ") + node::name ); return s; }\
	s = MHWRender::MDrawRegistry::deregisterDrawOverrideCreator( node::drawDbClassification, node::drawRegistrantId );\
	if( !s ) { s.perror( MString("Failed to deregister: ") + node::name ); return s; }\
}

