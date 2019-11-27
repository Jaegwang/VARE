//----------------------//
// BoraBgeoImporter.cpp //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2017.09.15                               //
//-------------------------------------------------------//

#include <BoraBgeoImporter.h>
#include <MayaCreateAttrib.h>
#include <BoraData.h>
#include <boost/regex.hpp>
#include <boost/filesystem.hpp>

MTypeId BoraBgeoImporter::id( 0x200002 );
MString BoraBgeoImporter::name("BoraBgeoImporter");
MString	BoraBgeoImporter::drawDbClassification( "drawdb/geometry/BoraBgeoImporter" );
MString	BoraBgeoImporter::drawRegistrantId( "BoraBgeoImporterNodePlugin" );

MObject BoraBgeoImporter::inTimeObj;
MObject BoraBgeoImporter::cachePathObj;
MObject BoraBgeoImporter::outDataObj;

BoraBgeoImporter::BoraBgeoImporter()
{
}

void*
BoraBgeoImporter::creator()
{
    return new BoraBgeoImporter();
}

void
BoraBgeoImporter::postConstructor()
{
	MPxNode::postConstructor();
    
    nodeObj = thisMObject();
    nodeFn.setObject( nodeObj );
    dagNodeFn.setObject( nodeObj );
    nodeFn.setName("BoraBgeoImporter#");    
}

MStatus
BoraBgeoImporter::initialize()
{
	MStatus s = MS::kSuccess;

	MFnUnitAttribute    uAttr;
	MFnEnumAttribute    eAttr;
	MFnTypedAttribute   tAttr;
	MFnNumericAttribute nAttr;

	CreateTimeAttr( uAttr, inTimeObj, "inTime", 0.0 );
	uAttr.setHidden(0); tAttr.setKeyable(1);
	CHECK_MSTATUS( addAttribute( inTimeObj ) );

	CreateStringAttr( tAttr, cachePathObj, "cachePath", "" );
	tAttr.setHidden(0); tAttr.setStorable(1); tAttr.setKeyable(1);
	CHECK_MSTATUS( addAttribute( cachePathObj ) );

	CreateCustomAttr( tAttr, outDataObj, "outData", BoraData );
    tAttr.setHidden(0); tAttr.setWritable(0); tAttr.setKeyable(0); tAttr.setStorable(0);
    CHECK_MSTATUS( addAttribute( outDataObj ) );
	
	attributeAffects( inTimeObj   , outDataObj );
	attributeAffects( cachePathObj, outDataObj );

	return s;
}

MStatus
BoraBgeoImporter::connectionMade( const MPlug& plug, const MPlug& otherPlug, bool asSrc )
{		
	if( is1stTime )
	{
		is1stTime = false;		
		nodeName = nodeFn.name();
		MString comm = MString("connectAttr time1.outTime ")+nodeName+MString(".inTime");

		MGlobal::executeCommand( comm );
	}
	
	return MPxNode::connectionMade( plug, otherPlug, asSrc );
}

MStatus
BoraBgeoImporter::compute( const MPlug& plug, MDataBlock& dataBlock )
{
	if( plug != outDataObj ) { return MS::kUnknownParameter; }

	dataBlockPtr = &dataBlock;
	nodeName = nodeFn.name();
	MThreadUtils::syncNumOpenMPThreads();

    int frame = dataBlock.inputValue( inTimeObj ).asTime().as( MTime::uiUnit() );
	const char* path = dataBlock.inputValue( cachePathObj ).asString().asChar();

 	//(Path)(Prefix)(PaddingNum)(Extension)
	boost::regex ex("(.+)/(.+\\D)(\\d+).(\\w+)");
	boost::cmatch m;
	boost::regex_match( path, m, ex );

	std::string pat( m[1].first, m[1].second );
	std::string pre( m[2].first, m[2].second );
	std::string pad( m[3].first, m[3].second );
	std::string ext( m[4].first, m[4].second );

	std::stringstream ss0;
	ss0 << pat << "/" << pre;
	ss0 << std::setfill('0') << std::setw(pad.length()) << frame;
	ss0 << "." << ext;

	std::stringstream ss1;
	ss1 << pat << "/" << pre << frame;
	ss1 << "." << ext;

	std::string filePath;
	if( boost::filesystem::exists( ss0.str() ))
	{
		filePath = ss0.str();
	}
	if( boost::filesystem::exists( ss1.str() ))
	{
		filePath = ss1.str();
	}


	ImportBgeoPointAttribV3( filePath.c_str(), "P", point_p );

	bgeoFile = filePath;	

	MFnPluginData dataFn;
	dataFn.create( BoraData::id );
	BoraData* newData = (BoraData*)dataFn.data();
	{
		newData->nodePtr = (void*)this;
		newData->nodeName = name;
		newData->nodeId = id;
	}

	MDataHandle outDataHnd = dataBlock.outputValue( outDataObj );
	outDataHnd.set( newData );

	dataBlock.setClean( plug );
	return MS::kSuccess;
}

void
BoraBgeoImporter::draw( M3dView& view, const MDagPath& path, M3dView::DisplayStyle style, M3dView::DisplayStatus status )
{
	view.beginGL();
	{
		draw();
	}
	view.endGL();
}

void
BoraBgeoImporter::draw( int drawingMode, const MHWRender::MDrawContext* context )
{
	glBegin( GL_POINTS );
	for( int n=0; n<point_p.num(); ++n )
	{
		const Vec3f& p = point_p[n];
		glVertex3f( p.x, p.y, p.z );
	}
	glEnd();
}

MBoundingBox
BoraBgeoImporter::boundingBox() const
{
	MBoundingBox bBox;
	bBox.expand( MPoint( -100.f, -100.f, -100.f ) );
	bBox.expand( MPoint(  100.f,  100.f,  100.f ) );
	
	return bBox;
}

bool
BoraBgeoImporter::isBounded() const
{
	return true;
}

