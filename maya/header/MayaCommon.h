//--------------//
// MayaCommon.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.02.22                               //
//-------------------------------------------------------//

#ifndef _MayaCommon_h_
#define _MayaCommon_h_

#include <Bora.h>

// To avoid conflicts between Cuda and Maya.
#define short2  MAYA_short2
#define short3  MAYA_short3
#define long2   MAYA_long2
#define long3   MAYA_long3
#define int2    MAYA_int2
#define int3    MAYA_int3
#define float2  MAYA_float2
#define float3  MAYA_float3
#define double2 MAYA_double2
#define double3 MAYA_double3
#define double4 MAYA_double4

#include <maya/MPlug.h>
#include <maya/MTime.h>
#include <maya/MPlane.h>
#include <maya/MTimer.h>
#include <maya/MItDag.h>
#include <maya/MImage.h>
#include <maya/MPoint.h>
#include <maya/MColor.h>
#include <maya/MGlobal.h>
#include <maya/MStatus.h>
#include <maya/MFnMesh.h>
#include <maya/MVector.h>
#include <maya/MMatrix.h>
#include <maya/M3dView.h>
#include <maya/MSyntax.h>
#include <maya/MCursor.h>
#include <maya/MPxData.h>
#include <maya/MFnField.h>
#include <maya/MDagPath.h>
#include <maya/MFnFluid.h>
#include <maya/MArgList.h>
#include <maya/MMaterial.h>
#include <maya/MUserData.h>
#include <maya/MDistance.h>
#include <maya/MFnCamera.h>
#include <maya/MIOStream.h>
#include <maya/MDrawData.h>
#include <maya/MPlugArray.h>
#include <maya/MItCurveCV.h>
#include <maya/MFnDagNode.h>
#include <maya/MToolsInfo.h>
#include <maya/MUiMessage.h>
#include <maya/MPxCommand.h>
#include <maya/MPxContext.h>
#include <maya/MDGMessage.h>
#include <maya/MHWGeometry.h>
#include <maya/MDGModifier.h>
#include <maya/MQuaternion.h>
#include <maya/MDataHandle.h>
#include <maya/MFnMeshData.h>
#include <maya/MFloatArray.h>
#include <maya/MItMeshEdge.h>
#include <maya/MPointArray.h>
#include <maya/MItGeometry.h>
#include <maya/MColorArray.h>
#include <maya/MRenderUtil.h>
#include <maya/MFileObject.h>
#include <maya/MDagModifier.h>
#include <maya/MAnimControl.h>
#include <maya/MFloatMatrix.h>
#include <maya/MNodeMessage.h>
#include <maya/MThreadUtils.h>
#include <maya/MComputation.h>
#include <maya/MMatrixArray.h>
#include <maya/MArgDatabase.h>
#include <maya/MFnTransform.h>
#include <maya/MDrawRequest.h>
#include <maya/MBoundingBox.h>
#include <maya/MDrawRequest.h>
#include <maya/MDrawContext.h>
#include <maya/MEventMessage.h>
#include <maya/MStateManager.h>
#include <maya/MDrawRegistry.h>
#include <maya/MFnStringData.h>
#include <maya/MModelMessage.h>
#include <maya/MFnNurbsCurve.h>
#include <maya/MDagPathArray.h>
#include <maya/MFnMatrixData.h>
#include <maya/MItMeshVertex.h>
#include <maya/MFnPluginData.h>
#include <maya/MFeedbackLine.h>
#include <maya/MShaderManager.h>
#include <maya/MRampAttribute.h>
#include <maya/MPxLocatorNode.h>
#include <maya/MItMeshPolygon.h>
#include <maya/MSelectionList.h>
#include <maya/MFnNumericData.h>
#include <maya/MEulerRotation.h>
#include <maya/MPxToolCommand.h> 
#include <maya/MSelectionMask.h>
#include <maya/MAttributeSpec.h>
#include <maya/MAttributeIndex.h>
#include <maya/MPxGeometryData.h>
#include <maya/MTextureManager.h>
#include <maya/MPxDrawOverride.h>
#include <maya/MProgressWindow.h>
#include <maya/MPxDeformerNode.h>
#include <maya/MFnIntArrayData.h>
#include <maya/MPxSurfaceShape.h>
#include <maya/MMeshIntersector.h>
#include <maya/MFnUnitAttribute.h>
#include <maya/MItSelectionList.h>
#include <maya/MFloatPointArray.h>
#include <maya/MFnEnumAttribute.h>
#include <maya/MGLFunctionTable.h>
#include <maya/MHardwareRenderer.h>
#include <maya/MPxManipContainer.h>
#include <maya/MFnPointArrayData.h>
#include <maya/MFnNurbsCurveData.h>
#include <maya/MPxSurfaceShapeUI.h>
#include <maya/MFnParticleSystem.h>
#include <maya/MFloatVectorArray.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MArrayDataBuilder.h>
#include <maya/MPxContextCommand.h>
#include <maya/MFnFloatArrayData.h>
#include <maya/MFnStringArrayData.h>
#include <maya/MFnMatrixAttribute.h>
#include <maya/MFnVectorArrayData.h>
#include <maya/MFnDoubleArrayData.h>
#include <maya/MViewport2Renderer.h>
#include <maya/MPxSubSceneOverride.h>
#include <maya/MAttributeSpecArray.h>
#include <maya/MPxGeometryOverride.h>
#include <maya/MFnMessageAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MPxGeometryIterator.h>
#include <maya/MFnCompoundAttribute.h>
#include <maya/MFnComponentListData.h>
#include <maya/MHWGeometryUtilities.h>
#include <maya/MTransformationMatrix.h>
#include <maya/MTextureEditorDrawInfo.h> 
#include <maya/MFnSingleIndexedComponent.h>

#if MAYA_API_VERSION >= 201602
#include <maya/MSelectionContext.h>
#include <maya/MPxComponentConverter.h>
#endif

using namespace std;
using namespace Bora;

#endif

