//--------------------//
// MayaViewOverride.h //
//-------------------------------------------------------//
// author: Jaegwwang Lim @ Dexter Studios                //
// last update: 2017.09.15                               //
//-------------------------------------------------------//

#ifndef _MayaViewOverride_h_
#define _MayaViewOverride_h_

#include <MayaCommon.h>

template <class T>
class MayaViewOverride : public MHWRender::MPxDrawOverride
{
	private:

		class ViewDrawData : public MUserData
		{
			public:

				void* dgNodePtr=0;
				MDagPath dgNodePath;

			public:

				ViewDrawData() : MUserData(false)
				{}
		};

	protected:

		T* dgNodePtr=0;

	public:

		MayaViewOverride( const MObject& );
		static MHWRender::MPxDrawOverride* Creator( const MObject& );
		virtual MHWRender::DrawAPI supportedDrawAPIs() const;
		virtual bool isBounded( const MDagPath&, const MDagPath& ) const;
		virtual MBoundingBox boundingBox( const MDagPath&, const MDagPath& ) const;
		virtual MUserData* prepareForDraw( const MDagPath&, const MDagPath&, const MHWRender::MFrameContext&, MUserData* );
		static void draw( const MHWRender::MDrawContext&, const MUserData* );
		virtual bool hasUIDrawables() const { return true; }
};

template <class T>
MayaViewOverride<T>::MayaViewOverride( const MObject& dgNodeObj )
: MHWRender::MPxDrawOverride( dgNodeObj, MayaViewOverride::draw )
{
	MStatus status = MS::kSuccess;
	MFnDependencyNode nodeFn( dgNodeObj, &status );

	if( status == MS::kSuccess ) {

		dgNodePtr = (T*)nodeFn.userNode();

	} else {

		dgNodePtr = (T*)NULL;

	}
}

template <class T>
MHWRender::MPxDrawOverride*
MayaViewOverride<T>::Creator( const MObject& dgNodeObj )
{
	return new MayaViewOverride( dgNodeObj );
}

template <class T>
MHWRender::DrawAPI
MayaViewOverride<T>::supportedDrawAPIs() const
{
	#if MAYA_API_VERSION >= 201650
	 return ( MHWRender::kOpenGL | MHWRender::kOpenGLCoreProfile );
	#else
	 return ( MHWRender::kOpenGL );
	#endif
}

template <class T>
bool
MayaViewOverride<T>::isBounded( const MDagPath& dgNodePath, const MDagPath& cameraPath ) const
{
	return true;
}

template <class T>
MBoundingBox
MayaViewOverride<T>::boundingBox( const MDagPath& dgNodePath, const MDagPath& cameraPath ) const
{
	return dgNodePtr->boundingBox();
}

template <class T>
MUserData*
MayaViewOverride<T>::prepareForDraw( const MDagPath& dgNodePath, const MDagPath& cameraPath, const MHWRender::MFrameContext& frameContext, MUserData* oldData )
{
	if( !dgNodePtr ) { return (MUserData*)NULL; }

	MStatus status = MS::kSuccess;

	MFnDependencyNode dgNodeFn( dgNodePath.node(), &status );
	if( !status ) { return (MUserData*)NULL; }

	ViewDrawData* drawData = dynamic_cast<ViewDrawData*>( oldData );
	if( !drawData ) { drawData = new ViewDrawData(); }

	drawData->dgNodePtr  = dgNodePtr;
	drawData->dgNodePath = dgNodePath;

	return (MUserData*)drawData;
}

template <class T>
void
MayaViewOverride<T>::draw( const MHWRender::MDrawContext& context, const MUserData* data )
{
	MHWRender::MRenderer* theRenderer = MHWRender::MRenderer::theRenderer();
	if( !theRenderer ) { return; }
	if( !theRenderer->drawAPIIsOpenGL() ) { return; }

	MHWRender::MStateManager* stateMgr = context.getStateManager();
	const ViewDrawData* drawDataPtr = dynamic_cast<const ViewDrawData*>( data );
	if( !stateMgr || !drawDataPtr ) { return; }
	const ViewDrawData& drawData = *drawDataPtr;

	T* dgNodePtr = (T*)( drawData.dgNodePtr );
	if( !dgNodePtr ) { return; }
	T& dgNode = *dgNodePtr;

	int drawingMode = 0;
	{
		MDAGDrawOverrideInfo drawOverrideInfo = drawData.dgNodePath.getDrawOverrideInfo();
		const MDAGDrawOverrideInfo& objectOverrideInfo = drawOverrideInfo;
		if( objectOverrideInfo.fOverrideEnabled && !objectOverrideInfo.fEnableVisible ) { return; }

		const bool overideTemplated = objectOverrideInfo.fOverrideEnabled && ( objectOverrideInfo.fDisplayType == MDAGDrawOverrideInfo::kDisplayTypeTemplate );
		const bool overrideNoShaded = objectOverrideInfo.fOverrideEnabled && !objectOverrideInfo.fEnableShading;

		if( overideTemplated ) {

			drawingMode = 3;

		} else if( overrideNoShaded ) {

			drawingMode = 3;

		} else {

			switch( context.getDisplayStyle() )
			{
				default:
				case MHWRender::MFrameContext::kWireFrame:     { drawingMode = 3; break; }
				case MHWRender::MFrameContext::kGouraudShaded: { drawingMode = 4; break; }
				case MHWRender::MFrameContext::kTextured:      { drawingMode = 5; break; }
			}

		}
	}

	const MMatrix worldViewMatrix  ( context.getMatrix( MHWRender::MFrameContext::kWorldViewMtx  ) );
	const MMatrix projectionMatrix ( context.getMatrix( MHWRender::MFrameContext::kProjectionMtx ) );

	glPushAttrib( GL_CURRENT_BIT );
	{
		glMatrixMode( GL_PROJECTION );
		glPushMatrix();
		glLoadMatrixd( projectionMatrix.matrix[0] );

		glMatrixMode( GL_MODELVIEW );
		glPushMatrix();
		glLoadMatrixd( worldViewMatrix.matrix[0] );
		{
			dgNode.draw( drawingMode, &context );
		}
		glMatrixMode( GL_MODELVIEW );
		glPopMatrix();

		glMatrixMode( GL_PROJECTION );
		glPopMatrix();
	}
	glPopAttrib();
}

#endif // MAYA_API_VERSION

