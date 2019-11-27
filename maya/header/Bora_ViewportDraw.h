//---------------------//
// Bora_ViewportDraw.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studstd::ios              //
// last update: 2017.11.21                               //
//-------------------------------------------------------//

#ifndef _Bora_ViewportDraw_h_
#define _Bora_ViewportDraw_h_

#include <MayaCommon.h>

// don't delete after draw
class Bora_DrawData : public MUserData
{
	public:

		void*    dgNodePtr;
		MDagPath dgNodePath;

	public:

		Bora_DrawData()
		: MUserData(false)
		{
			dgNodePtr = (void*)NULL;
		}
};

// constructor
// It is to be called once by Maya at the first time of changing into Viewport 2.0.

// Creator()
// It creates an instance of the draw override class.

// supportedDrawAPIs()
// It returns a value which is formed as the bitwise or of MHWRender::DrawAPI elements to indicate that the override supports multiple draw APIs.

// isBounded()
// It it to be called by Maya to determine if the drawable object is bounded or not.
// It can return true or false to indicate whether the object is bounded or not respectively.
// If the object is not bounded then it will never be culled by the current camera frustum used for drawing.
// In this situation the boundingBox() method will not be called since the bounding box is not required.

// boundingBox()
// It is to be called by Maya whenever the bounding box of the drawable object is needed.
// It should always return the object space bounding box for whatever is to be drawn.
// If the bounding box is incorrect the node may be culled at the wrong time and the custom draw method might not be called.

// prepareForDraw()
// It is to be called by Maya each time the object needs to be drawn.
// Any data needed from the Maya dependency graph must be retrieved and cached in this stage.
// It is invalid to pull data from the Maya dependency graph in the draw callback method and Maya may become unstable if that is attempted.
// Implementors may allow Maya to handle the data caching by returning a pointer to the data from this method.
// The pointer must be to a class derived from MUserData.
// This same pointer will be passed to the draw callback.
// On subsequent draws, the pointer will also be passed back into this method so that the data may be modified and reused instead of reallocated.
// If a different pointer is returned Maya will delete the old data.
// If the cache should not be maintained between draws, set the delete after use flag on the user data.
// In all cases, the lifetime and ownership of the user data is handled by Maya and the user should not try to delete the data themselves.
// Data caching occurs per-instance of the associated DAG object.
// The lifetime of the user data can be longer than the associated node, instance or draw override.
// Due to internal caching, the user data can be deleted after an arbitrary long time.
// One should therefore be careful to not access stale objects from the user data destructor.
// If it is not desirable to allow Maya to handle data caching, simply return NULL in this method and ignore the user data parameter in the draw callback method.

// draw()
// User draw callback definition, draw context and blind user data are parameters.

template <class T>
class Bora_DrawOverride : public MHWRender::MPxDrawOverride
{
	protected:

		T* dgNodePtr;

	public:

		Bora_DrawOverride( const MObject& );
		static MHWRender::MPxDrawOverride* Creator( const MObject& );
		virtual MHWRender::DrawAPI supportedDrawAPIs() const;
		virtual bool isBounded( const MDagPath&, const MDagPath& ) const;
		virtual MBoundingBox boundingBox( const MDagPath&, const MDagPath& ) const;
		virtual MUserData* prepareForDraw( const MDagPath&, const MDagPath&, const MHWRender::MFrameContext&, MUserData* );
		static void draw( const MHWRender::MDrawContext&, const MUserData* );
		virtual bool hasUIDrawables() const { return true; }
};

template <class T>
Bora_DrawOverride<T>::Bora_DrawOverride( const MObject& dgNodeObj )
: MHWRender::MPxDrawOverride( dgNodeObj, Bora_DrawOverride::draw )
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
Bora_DrawOverride<T>::Creator( const MObject& dgNodeObj )
{
	return new Bora_DrawOverride( dgNodeObj );
}

template <class T>
MHWRender::DrawAPI
Bora_DrawOverride<T>::supportedDrawAPIs() const
{
	#if MAYA_API_VERSION >= 201650
	 return ( MHWRender::kOpenGL | MHWRender::kOpenGLCoreProfile );
	#else
	 return ( MHWRender::kOpenGL );
	#endif
}

template <class T>
bool
Bora_DrawOverride<T>::isBounded( const MDagPath& dgNodePath, const MDagPath& cameraPath ) const
{
	return true;
}

template <class T>
MBoundingBox
Bora_DrawOverride<T>::boundingBox( const MDagPath& dgNodePath, const MDagPath& cameraPath ) const
{
	return dgNodePtr->boundingBox();
}

template <class T>
MUserData*
Bora_DrawOverride<T>::prepareForDraw( const MDagPath& dgNodePath, const MDagPath& cameraPath, const MHWRender::MFrameContext& frameContext, MUserData* oldData )
{
	if( !dgNodePtr ) { return (MUserData*)NULL; }

	MStatus status = MS::kSuccess;

	MFnDependencyNode dgNodeFn( dgNodePath.node(), &status );
	if( !status ) { return (MUserData*)NULL; }

	Bora_DrawData* drawData = dynamic_cast<Bora_DrawData*>( oldData );
	if( !drawData ) { drawData = new Bora_DrawData(); }

	drawData->dgNodePtr  = dgNodePtr;
	drawData->dgNodePath = dgNodePath;

	return (MUserData*)drawData;
}

template <class T>
void
Bora_DrawOverride<T>::draw( const MHWRender::MDrawContext& context, const MUserData* data )
{
	MHWRender::MRenderer* theRenderer = MHWRender::MRenderer::theRenderer();
	if( !theRenderer ) { return; }
	if( !theRenderer->drawAPIIsOpenGL() ) { return; }

	MHWRender::MStateManager* stateMgr = context.getStateManager();
	const Bora_DrawData* drawDataPtr = dynamic_cast<const Bora_DrawData*>( data );
	if( !stateMgr || !drawDataPtr ) { return; }
	const Bora_DrawData& drawData = *drawDataPtr;

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
			dgNode.draw( drawingMode );
		}
		glMatrixMode( GL_MODELVIEW );
		glPopMatrix();

		glMatrixMode( GL_PROJECTION );
		glPopMatrix();
	}
	glPopAttrib();
}

#endif // MAYA_API_VERSION

