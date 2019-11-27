//-------------//
// MayaUtils.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
//         Jaegwang Lim @ Dexter Studios                 //
// last update: 2019.01.23                               //
//-------------------------------------------------------//

#pragma once
#include <Bora.h>
#include <MayaCommon.h>

#define MAYA_CHECK_ERROR(s,msg) if(s!=MS::kSuccess) { MGlobal::displayError(msg); return s; }

template <typename T>
inline MPoint AsMPoint( const Vector3<T>& v )
{
	return MPoint( v.x, v.y, v.z );
}

template <typename T>
inline MMatrix AsMMatrix( const Matrix44<T>& m )
{
    MMatrix tmp;

	for( int i=0; i<4; ++i )
    for( int j=0; j<4; ++j )
    {{
        tmp(i,j) = (double)m(j,i);
    }}

    return tmp;
}

inline Mat44f AsMat44f( const MMatrix& m )
{
    Mat44f tmp;

	for( int i=0; i<4; ++i )
    for( int j=0; j<4; ++j )
    {{
        tmp(i,j) = (float)m(j,i);
    }}

    return tmp;
}

inline Mat44d AsMat44d( const MMatrix& m )
{
    Mat44d tmp;

	for( int i=0; i<4; ++i )
    for( int j=0; j<4; ++j )
    {{
        tmp(i,j) = m(j,i);
    }}

    return tmp;
}

MObject NodeNameToMObject( const MString& nodeName );

bool GetWorldMatrix( const MObject& dagNodeObj, MMatrix& worldMatrix );

// p -> p (p=M*p)
void ApplyXForm( MPoint& p, const MMatrix& M );

// p -> q (q=M*p)
void ApplyXForm( const MMatrix& M, const MPoint& p, MPoint& q );

void Copy( Vec3fArray& to, const MPointArray& from );

void Copy( Vec3fArray& to, const MVectorArray& from );

bool Convert( TriangleMesh& mesh, MObject& meshObj, bool vPosOnly=false, const char* uvSetName=NULL );

float CurrentFPS();

inline bool GetConnectedNodeObject( const MObject& nodeObj, const MObject& plgObj, bool asDestination, MObject& connectedNodeObj )
{
    MFnDependencyNode dgFn( nodeObj );
    MPlug plg = dgFn.findPlug( plgObj );
    if( plg.isConnected() )
    {
        MPlugArray connectedPlgs;
        if( asDestination ) { plg.connectedTo( connectedPlgs, true, false ); }
        else                { plg.connectedTo( connectedPlgs, false, true ); }
        
        connectedNodeObj = connectedPlgs[0].node();
        
        return true;
    }

    return false;
}

