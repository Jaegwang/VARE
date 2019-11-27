//--------------------//
// MayaCreateAttrib.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2017.09.20                               //
//-------------------------------------------------------//

#ifndef _MayaCreateAttrib_h_
#define _MayaCreateAttrib_h_

#include <MayaCommon.h>

inline bool
CreateTimeAttr( MFnUnitAttribute& uAttr, MObject& attrObj, const MString& longName, double initValue )
{
	MStatus status;
	attrObj = uAttr.create( longName, longName, MFnUnitAttribute::kTime, initValue, &status );
	MAYA_CHECK_ERROR( status, MString("FAILED to create") + "'." + longName + "' attribute" );
	return false;
}

inline bool
CreateAngleAttr( MFnUnitAttribute& uAttr, MObject& attrObj, const MString& longName, double initValue )
{
	MStatus status;
	attrObj = uAttr.create( longName, longName, MFnUnitAttribute::kAngle, initValue, &status );
	MAYA_CHECK_ERROR( status, MString("FAILED to create") + "'." + longName + "' attribute" );
	return false;
}

inline bool
CreateBoolAttr( MFnNumericAttribute& nAttr, MObject& attrObj, const MString& longName, bool initValue )
{
	MStatus status;
	attrObj = nAttr.create( longName, longName, MFnNumericData::kBoolean, initValue, &status );
	MAYA_CHECK_ERROR( status, MString("FAILED to create") + "'." + longName + "' attribute" );
	return false;
}

inline bool
CreateIntAttr( MFnNumericAttribute& nAttr, MObject& attrObj, const MString& longName, int initValue )
{
	MStatus status;
	attrObj = nAttr.create( longName, longName, MFnNumericData::kInt, initValue, &status );
	MAYA_CHECK_ERROR( status, MString("FAILED to create") + "'." + longName + "' attribute" );
	return false;
}

inline bool
CreateShortAttr( MFnNumericAttribute& nAttr, MObject& attrObj, const MString& longName, short initValue )
{
	MStatus status;
	attrObj = nAttr.create( longName, longName, MFnNumericData::kShort, initValue, &status );
	MAYA_CHECK_ERROR( status, MString("FAILED to create") + "'." + longName + "' attribute" );
	return false;
}

inline bool
CreateFloatAttr( MFnNumericAttribute& nAttr, MObject& attrObj, const MString& longName, float initValue )
{
	MStatus status;
	attrObj = nAttr.create( longName, longName, MFnNumericData::kFloat, initValue, &status );
	MAYA_CHECK_ERROR( status, MString("FAILED to create") + "'." + longName + "' attribute" );
	return false;
}

inline bool
CreateDoubleAttr( MFnNumericAttribute& nAttr, MObject& attrObj, const MString& longName, double initValue )
{
	MStatus status;
	attrObj = nAttr.create( longName, longName, MFnNumericData::kDouble, initValue, &status );
	MAYA_CHECK_ERROR( status, MString("FAILED to create") + "'." + longName + "' attribute" );
	return false;
}

inline bool
CreatePointAttr( MFnNumericAttribute& nAttr, MObject& attrObj, const MString& longName )
{
	MStatus status;
	attrObj = nAttr.createPoint( longName, longName, &status );
	MAYA_CHECK_ERROR( status, MString("FAILED to create") + "'." + longName + "' attribute" );
	return false;
}

inline bool
CreateI2Attr( MFnNumericAttribute& nAttr, MObject& attrObj, const MString& longName, int initValue )
{
	MStatus status;
	attrObj = nAttr.create( longName, longName, MFnNumericData::k2Int, initValue, &status );
	MAYA_CHECK_ERROR( status, MString("FAILED to create") + "'." + longName + "' attribute" );
	return false;
}

inline bool
CreateI3Attr( MFnNumericAttribute& nAttr, MObject& attrObj, const MString& longName, int initValue )
{
	MStatus status;
	attrObj = nAttr.create( longName, longName, MFnNumericData::k3Int, initValue, &status );
	MAYA_CHECK_ERROR( status, MString("FAILED to create") + "'." + longName + "' attribute" );
	return false;
}

inline bool
CreateF2Attr( MFnNumericAttribute& nAttr, MObject& attrObj, const MString& longName, float initValue )
{
	MStatus status;
	attrObj = nAttr.create( longName, longName, MFnNumericData::k2Float, initValue, &status );
	MAYA_CHECK_ERROR( status, MString("FAILED to create") + "'." + longName + "' attribute" );
	return false;
}

inline bool
CreateF3Attr( MFnNumericAttribute& nAttr, MObject& attrObj, const MString& longName, float initValue )
{
	MStatus status;
	attrObj = nAttr.create( longName, longName, MFnNumericData::k3Float, initValue, &status );
	MAYA_CHECK_ERROR( status, MString("FAILED to create") + "'." + longName + "' attribute" );
	return false;
}

inline bool
CreateD2Attr( MFnNumericAttribute& nAttr, MObject& attrObj, const MString& longName, double initValue )
{
	MStatus status;
	attrObj = nAttr.create( longName, longName, MFnNumericData::k2Double, initValue, &status );
	MAYA_CHECK_ERROR( status, MString("FAILED to create") + "'." + longName + "' attribute" );
	return false;
}

inline bool
CreateD3Attr( MFnNumericAttribute& nAttr, MObject& attrObj, const MString& longName, double initValue )
{
	MStatus status;
	attrObj = nAttr.create( longName, longName, MFnNumericData::k3Double, initValue, &status );
	MAYA_CHECK_ERROR( status, MString("FAILED to create") + "'." + longName + "' attribute" );
	return false;
}

inline bool
CreateD4Attr( MFnNumericAttribute& nAttr, MObject& attrObj, const MString& longName, double initValue )
{
	MStatus status;
	attrObj = nAttr.create( longName, longName, MFnNumericData::k4Double, initValue, &status );
	MAYA_CHECK_ERROR( status, MString("FAILED to create") + "'." + longName + "' attribute" );
	return false;
}

inline bool
CreateEnumAttr( MFnEnumAttribute& eAttr, MObject& attrObj, const MString& longName, int initValue )
{
	MStatus status;
	attrObj = eAttr.create( longName, longName, initValue );
	MAYA_CHECK_ERROR( status, MString("FAILED to create") + "'." + longName + "' attribute" );
	return false;
}

inline bool
CreateStringAttr( MFnTypedAttribute& tAttr, MObject& attrObj, const MString& longName, const MString str )
{
	MStatus status;
	MFnStringData strDataFn;
	MObject defaultStrObj( strDataFn.create( str ) );
	attrObj = tAttr.create( longName, longName, MFnData::kString, defaultStrObj, &status );
	MAYA_CHECK_ERROR( status, MString("FAILED to create") + "'." + longName + "' attribute" );
	return false;
}

inline bool
CreateColorAttr( MFnNumericAttribute& nAttr, MObject& attrObj, const MString& longName )
{
	MStatus status;
	attrObj = nAttr.createColor( longName, longName, &status );
	MAYA_CHECK_ERROR( status, MString("FAILED to create") + "'." + longName + "' attribute" );
	return false;
}

inline bool
CreateCurveAttr( MFnTypedAttribute& tAttr, MObject& attrObj, const MString& longName )
{
	MStatus status;
	attrObj = tAttr.create( longName, longName, MFnData::kNurbsCurve, &status );
	MAYA_CHECK_ERROR( status, MString("FAILED to create") + "'." + longName + "' attribute" );
	return false;
}

inline bool
CreateCurveDataAttr( MFnTypedAttribute& tAttr, MObject& attrObj, const MString& longName )
{
	MStatus status;
	attrObj = tAttr.create( longName, longName, MFnNurbsCurveData::kNurbsCurve, &status );
	MAYA_CHECK_ERROR( status, MString("FAILED to create") + "'." + longName + "' attribute" );
	return false;
}

inline bool
CreateMeshAttr( MFnTypedAttribute& tAttr, MObject& attrObj, const MString& longName )
{
	MStatus status;
	attrObj = tAttr.create( longName, longName, MFnData::kMesh, &status );
	MAYA_CHECK_ERROR( status, MString("FAILED to create") + "'." + longName + "' attribute" );
	return false;
}

inline bool
CreateIArrayAttr( MFnTypedAttribute& tAttr, MObject& attrObj, const MString& longName )
{
	MStatus status;
	attrObj = tAttr.create( longName, longName, MFnData::kIntArray, &status );
	MAYA_CHECK_ERROR( status, MString("FAILED to create") + "'." + longName + "' attribute" );
	return false;
}

inline bool
CreateFArrayAttr( MFnTypedAttribute& tAttr, MObject& attrObj, const MString& longName )
{
	MStatus status;
	attrObj = tAttr.create( longName, longName, MFnData::kFloatArray, &status );
	MAYA_CHECK_ERROR( status, MString("FAILED to create") + "'." + longName + "' attribute" );
	return false;
}

inline bool
CreateDArrayAttr( MFnTypedAttribute& tAttr, MObject& attrObj, const MString& longName )
{
	MStatus status;
	attrObj = tAttr.create( longName, longName, MFnData::kDoubleArray, &status );
	MAYA_CHECK_ERROR( status, MString("FAILED to create") + "'." + longName + "' attribute" );
	return false;
}

inline bool
CreatePArrayAttr( MFnTypedAttribute& tAttr, MObject& attrObj, const MString& longName )
{
	MStatus status;
	attrObj = tAttr.create( longName, longName, MFnData::kPointArray, &status );
	MAYA_CHECK_ERROR( status, MString("FAILED to create") + "'." + longName + "' attribute" );
	return false;
}

inline bool
CreateVArrayAttr( MFnTypedAttribute& tAttr, MObject& attrObj, const MString& longName )
{
	MStatus status;
	attrObj = tAttr.create( longName, longName, MFnData::kVectorArray, &status );
	MAYA_CHECK_ERROR( status, MString("FAILED to create") + "'." + longName + "' attribute" );
	return false;
}

inline bool
CreateMatrixAttr( MFnMatrixAttribute& mAttr, MObject& attrObj, const MString& longName )
{
	MStatus status;
	attrObj = mAttr.create( longName, longName, MFnMatrixAttribute::kDouble, &status );
	MAYA_CHECK_ERROR( status, MString("FAILED to create") + "'." + longName + "' attribute" );
	return false;
}

inline bool
CreateSArrayAttr( MFnTypedAttribute& tAttr, MObject& attrObj, const MString& longName )
{
	MStatus status;
	attrObj = tAttr.create( longName, longName, MFnData::kStringArray, attrObj, &status );
	MAYA_CHECK_ERROR( status, MString("FAILED to create") + "'." + longName + "' attribute" );
	return false;
}

inline bool
CreateCompoundAttr( MFnCompoundAttribute& cAttr, MObject& attrObj, const MString& longName )
{
	MStatus status;
	attrObj = cAttr.create( longName, longName, &status );
	MAYA_CHECK_ERROR( status, MString("FAILED to create") + "'." + longName + "' attribute" );
	return false;
}

inline bool
CreateMessageAttr( MFnMessageAttribute& gAttr, MObject& attrObj, const MString& longName )
{
	MStatus status;
	attrObj = gAttr.create( longName, longName, &status );
	MAYA_CHECK_ERROR( status, MString("FAILED to create") + "'." + longName + "' attribute" );
	return false;
}

inline bool
CreateVectorAttr( MFnNumericAttribute& nAttr, MObject& attrObj, const MString& longName, MObject& xObj, MObject& yObj, MObject& zObj )
{
	MStatus status;
	attrObj = nAttr.create( longName, longName, xObj, yObj, zObj, &status );
	MAYA_CHECK_ERROR( status, MString("FAILED to create") + "'." + longName + "' attribute" );
	return false;
}

inline bool
Create3TupleAttr( MFnNumericAttribute& nAttr, MObject& attrObj, const MString& longName, MObject& xObj, MObject& yObj, MObject& zObj )
{
	MStatus status;
	attrObj = nAttr.create( longName, longName, xObj, yObj, zObj, &status );
	MAYA_CHECK_ERROR( status, MString("FAILED to create") + "'." + longName + "' attribute" );
	return false;
}

inline bool
CreateCurveRampAttr( MRampAttribute& rAttr, MObject& attrObj, const MString& longName )
{
	MStatus status;
	attrObj = rAttr.createCurveRamp( longName, longName, &status );
	MAYA_CHECK_ERROR( status, MString("FAILED to create") + "'." + longName + "' attribute" );
	return false;
}

inline bool
CreateColorRampAttr( MRampAttribute& rAttr, MObject& attrObj, const MString& longName )
{
	MStatus status;
	attrObj = rAttr.createColorRamp( longName, longName, &status );
	MAYA_CHECK_ERROR( status, MString("FAILED to create") + "'." + longName + "' attribute" );
	return false;
}

// data
#define CreateCustomAttr( tAttr, attrObj, longName, class )\
{\
	attrObj = tAttr.create( longName, longName, class::id, MObject::kNullObj );\
}

#endif

