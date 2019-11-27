//-----------------------//
// BoraVDBFromPoints.cpp //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2017.09.20                               //
//-------------------------------------------------------//

#include <BoraVDBFromPoints.h>
#include <MayaCreateAttrib.h>
#include <MayaRmanAttrib.h>
#include <BoraBgeoImporter.h>
#include <BoraData.h>

MTypeId BoraVDBFromPoints::id( 0x200003 );
MString BoraVDBFromPoints::name("BoraVDBFromPoints");
MString	BoraVDBFromPoints::drawDbClassification( "drawdb/geometry/BoraVDBFromPoints" );
MString	BoraVDBFromPoints::drawRegistrantId( "BoraVDBFromPointsNodePlugin" );

std::map<std::string, BoraVDBFromPoints*> BoraVDBFromPoints::instances;

MObject BoraVDBFromPoints::radiusFreqObj;
MObject BoraVDBFromPoints::radiusOffsetObj;
MObject BoraVDBFromPoints::radiusScaleObj;
MObject BoraVDBFromPoints::densityFreqObj;
MObject BoraVDBFromPoints::densityOffsetObj;
MObject BoraVDBFromPoints::densityScaleObj;
MObject BoraVDBFromPoints::radiusObj;
MObject BoraVDBFromPoints::voxelSizeObj;
MObject BoraVDBFromPoints::inDataObj;
MObject BoraVDBFromPoints::outputObj;

BoraVDBFromPoints::BoraVDBFromPoints()
{
}

void*
BoraVDBFromPoints::creator()
{
    return new BoraVDBFromPoints();
}

void
BoraVDBFromPoints::postConstructor()
{
	MPxNode::postConstructor();
    
    nodeObj = thisMObject();
    nodeFn.setObject( nodeObj );
    dagNodeFn.setObject( nodeObj );
    nodeFn.setName("BoraVDBFromPoints#");    
}

MStatus
BoraVDBFromPoints::initialize()
{
	MStatus s = MS::kSuccess;

	MFnUnitAttribute    uAttr;
	MFnEnumAttribute    eAttr;
	MFnTypedAttribute   tAttr;
    MFnNumericAttribute nAttr;
 
	CreateCustomAttr( tAttr, inDataObj, "inData", BoraData );
	nAttr.setHidden(0); nAttr.setStorable(0); nAttr.setKeyable(1);
	CHECK_MSTATUS( addAttribute( inDataObj ) );

	CreateFloatAttr( nAttr, radiusObj, "radius", 1.f );
	CHECK_MSTATUS( addAttribute( radiusObj ) );

	CreateFloatAttr( nAttr, voxelSizeObj, "voxelSize", 0.1f );
	CHECK_MSTATUS( addAttribute( voxelSizeObj ) );

    CreateFloatAttr( nAttr, outputObj, "output", 0.f );
    nAttr.setHidden(1); nAttr.setStorable(0);
    CHECK_MSTATUS( addAttribute( outputObj ) );
	

	attributeAffects( inDataObj   , outputObj );
	attributeAffects( radiusObj   , outputObj );
	attributeAffects( voxelSizeObj, outputObj );


	MDGMessage::addNodeRemovedCallback( BoraVDBFromPoints::destructor );	
	return s;
}

void
BoraVDBFromPoints::destructor( MObject& node, void* data )
{
	MStatus status = MS::kSuccess;
	MFnDependencyNode nodeFn( node, & status );

	if( nodeFn.typeId() == BoraVDBFromPoints::id )
	{
		BoraVDBFromPoints::instances.erase( nodeFn.name().asChar() );
	}
}

MStatus
BoraVDBFromPoints::connectionMade( const MPlug& plug, const MPlug& otherPlug, bool asSrc )
{
	if( is1stTime )
	{
		is1stTime = false;				
		MDGModifier mod;
		{
			MFnDagNode parentXFormDagFn( nodeObj );
			MObject parent = parentXFormDagFn.parent( 0 );
			parentXFormDagFn.setObject( parent );

			MPlug toPlg = parentXFormDagFn.findPlug( "dynamics" );
			MPlug fromPlg = MPlug( nodeObj, outputObj );

			if( !fromPlg.isConnected() ) mod.connect( fromPlg, toPlg );
		}
		mod.doIt();

		BoraVDBFromPoints::instances[ nodeFn.name().asChar() ] = this;

		std::stringstream ss;		
		ss << "BoraVDBFromPointsCmd " << nodeFn.name().asChar();

		CreateRmanAttrib( nodeFn.name().asChar(), "customShadingGroup", "" );
		CreateRmanAttrib( nodeFn.name().asChar(), "postShapeScript", ss.str().c_str() );
	}
	
	return MPxNode::connectionMade( plug, otherPlug, asSrc );
}

MStatus
BoraVDBFromPoints::compute( const MPlug& plug, MDataBlock& dataBlock )
{
	if( plug != outputObj ) { return MS::kUnknownParameter; }

	dataBlockPtr = &dataBlock;
	nodeName = nodeFn.name();
    MThreadUtils::syncNumOpenMPThreads();

	BoraData* boraData = (BoraData*)dataBlock.inputValue( inDataObj ).asPluginData();

	radius = dataBlock.inputValue( radiusObj ).asFloat();
	voxelSize = dataBlock.inputValue( voxelSizeObj ).asFloat();

	if( boraData->nodeName == BoraBgeoImporter::name )
	{
		BoraBgeoImporter* importer = (BoraBgeoImporter*)boraData->nodePtr;

		file = importer->bgeoFile;		
		
		points = &(importer->point_p);
		velocities = &(importer->point_v);
	}

	dataBlock.outputValue( outputObj ).setInt( 0 );
	dataBlock.setClean( plug );
	return MS::kSuccess;
}

void
BoraVDBFromPoints::draw( M3dView& view, const MDagPath& path, M3dView::DisplayStyle style, M3dView::DisplayStatus status )
{
	view.beginGL();
	{
		draw();
	}
	view.endGL();
}

void
BoraVDBFromPoints::draw( int drawingMode, const MHWRender::MDrawContext* context )
{
}

MBoundingBox
BoraVDBFromPoints::boundingBox() const
{
	MBoundingBox bBox;
	bBox.expand( MPoint(-20.f, -20.f, -20.f));
	bBox.expand( MPoint( 20.f,  20.f,  20.f));
	return bBox;
}

bool
BoraVDBFromPoints::isBounded() const
{
	return true;
}

