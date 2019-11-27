//---------------//
// MayaUtils.cpp //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studstd::ios              //
// last update: 2018.09.05                               //
//-------------------------------------------------------//

#include <MayaUtils.h>

MObject NodeNameToMObject( const MString& nodeName )
{
	MObject obj;
	MSelectionList sList;
	MStatus stat = MGlobal::getSelectionListByName( nodeName, sList );
	if( !stat ) { return MObject::kNullObj; }
	sList.getDependNode( 0, obj );
	return obj;
}

// dag node object of a transform node -> world matrix
bool GetWorldMatrix( const MObject& dagNodeObj, MMatrix& worldMat )
{
	MStatus status = MS::kSuccess;

	MFnDagNode dagFn( dagNodeObj, &status );
	if( status != MS::kSuccess ) { return false; } // Don't print any error messages here!

	MDagPath dagPath;
	status = dagFn.getPath( dagPath );
	if( status != MS::kSuccess ) { return false; } // Don't print any error messages here!

	worldMat = dagPath.inclusiveMatrix();

	return true;
}

void ApplyXForm( MPoint& p, const MMatrix& M )
{
	const double (*matrix)[4] = M.matrix;
	const double x=p.x, y=p.y, z=p.z;

	p.x = matrix[0][0]*x + matrix[1][0]*y + matrix[2][0]*z + matrix[3][0];
	p.y = matrix[0][1]*x + matrix[1][1]*y + matrix[2][1]*z + matrix[3][1];
	p.z = matrix[0][2]*x + matrix[1][2]*y + matrix[2][2]*z + matrix[3][2];
}

void ApplyXForm( const MMatrix& M, const MPoint& p, MPoint& q )
{
	const double (*matrix)[4] = M.matrix;
	const double &x=p.x, &y=p.y, &z=p.z;

	q.x = matrix[0][0]*x + matrix[1][0]*y + matrix[2][0]*z + matrix[3][0];
	q.y = matrix[0][1]*x + matrix[1][1]*y + matrix[2][1]*z + matrix[3][1];
	q.z = matrix[0][2]*x + matrix[1][2]*y + matrix[2][2]*z + matrix[3][2];
}

void Copy( Vec3fArray& to, const MPointArray& from )
{
	const unsigned int n = from.length();
	if(!n) { to.clear(); return; }

	to.resize( n );

	for( unsigned int i=0; i<n; ++i )
	{
		Vec3f& q = to[i];
		const MPoint& p = from[i];

		q.x = (float)p.x;
		q.y = (float)p.y;
		q.z = (float)p.z;
	}
}

void Copy( Vec3fArray& to, const MVectorArray& from )
{
	const unsigned int n = from.length();
	if(!n) { to.clear(); return; }

	to.resize( n );

	for( unsigned int i=0; i<n; ++i )
	{
		Vec3f& q = to[i];
		const MVector& p = from[i];

		q.x = (float)p.x;
		q.y = (float)p.y;
		q.z = (float)p.z;
	}
}

bool Convert( TriangleMesh& mesh, MObject& meshObj, bool vPosOnly, const char* uvSetName )
{
    if( meshObj.isNull() ) // no mesh object
    {
        mesh.clear();
        return true;
    }

	if( !vPosOnly ) // assumption: different topology
    {
        mesh.clear();
    }

	MStatus status = MS::kSuccess;

	MFnMesh meshFn( meshObj, &status );

	if( status != MS::kSuccess )
	{
		MGlobal::displayError( "Error@Convert(): Failed to get MFnMesh." );
		mesh.reset();
		return false;
	}

	MItMeshVertex vItr( meshObj, &status );
	if( status != MS::kSuccess )
	{
		MGlobal::displayError( "Error@Convert(): Failed to get MItMeshVertex." );
		mesh.reset();
		return false;
	}

	MItMeshPolygon fItr( meshObj, &status );
	if( status != MS::kSuccess )
	{
		MGlobal::displayError( "Error@Convert(): Failed to get MItMeshPolygon." );
		mesh.reset();
		return false;
	}

	const unsigned int numVertices = meshFn.numVertices();
    const unsigned int numPolygons = meshFn.numPolygons();

	//////////////////////
	// vertex positions //

	int         vIdx = 0;
	MPoint      localPos;
	MMatrix     localToWorld;
	MPointArray vP( numVertices ); // vertex positions

	if( GetWorldMatrix( meshObj, localToWorld ) )
	{
		for( vItr.reset(); !vItr.isDone(); vItr.next(), ++vIdx )
		{
			localPos = vItr.position( MSpace::kObject );
			ApplyXForm( localToWorld, localPos, vP[vIdx] );
		}
	}
	else
	{
		for( vItr.reset(); !vItr.isDone(); vItr.next(), ++vIdx )
		{
			vP[vIdx] = vItr.position( MSpace::kWorld );
		}
	}

	Copy( mesh.position, vP ); // mesh.position <- vP

	if( vPosOnly ) { return true; }

	/////////////////
	// UV-set name //

	bool toConvertUV = true;
	MString uvSetNameStr;
	MFloatArray vU, vV;

	if( !meshFn.numUVSets() || !meshFn.numUVs() )
	{
		toConvertUV = false;
	}
	else
	{
		MString inUVSetName( uvSetName );

		if( inUVSetName.length() == 0 )
		{
			toConvertUV = false;
		}
		else if( inUVSetName == MString("currentUVSet") )
		{
			uvSetNameStr = meshFn.currentUVSetName();
		}
		else
		{
			MStringArray uvSetNames;
			meshFn.getUVSetNames( uvSetNames );
			const unsigned int numUVSets = uvSetNames.length();

			for( unsigned int i=0; i<numUVSets; ++i )
			{
				if( inUVSetName == uvSetNames[i] )
				{
					uvSetNameStr = inUVSetName;
					break;
				}
			}
		}
	}

	if( toConvertUV )
	{
		if( !meshFn.getUVs( vU, vV, &uvSetNameStr ) ) { toConvertUV = false; }
	}

	//////////////////////
	// triangle indices //

    IndexArray& indices = mesh.indices;
    indices.reserve( numPolygons*2 );

    for( fItr.reset(); !fItr.isDone(); fItr.next() )
    {
        MIntArray vList;
        fItr.getVertices( vList );

        const unsigned int vCount = vList.length();

        if( vCount < 3 ) // invalid case
        {
            continue;
        }

        for( unsigned int i=0; i<vCount-2; ++i )
        {
            indices.append( vList[0]   );
            indices.append( vList[i+1] );
            indices.append( vList[i+2] );
        }
    }

	Vec3fArray& uvw = mesh.uvw;
	uvw.reserve( numPolygons*2 );

	if( toConvertUV )
	{
		for( fItr.reset(); !fItr.isDone(); fItr.next() )
		{
            MIntArray vList;
            fItr.getVertices( vList );

            const unsigned int vCount = vList.length();

            if( vCount < 3 ) // invalid case
            {
                continue;
            }

            MIntArray uvIndices;
			uvIndices.setLength( vCount );

			for( unsigned int i=0; i<vCount; ++i )
            {
                fItr.getUVIndex( i, uvIndices[i], &uvSetNameStr );
            }

            for( unsigned int i=0; i<vCount-2; ++i )
			{
				uvw.append( Vec3f( vU[uvIndices[0  ]], vV[uvIndices[0  ]], 0.f ) );
				uvw.append( Vec3f( vU[uvIndices[i+1]], vV[uvIndices[i+1]], 0.f ) );
				uvw.append( Vec3f( vU[uvIndices[i+2]], vV[uvIndices[i+2]], 0.f ) );
            }
		}
	}

	return true;
}


float CurrentFPS()
{
    float fps = 24.0f;

    MTime::Unit unit = MTime::uiUnit();

    if( unit != MTime::kInvalid )
    {
        MTime time( 1.0, MTime::kSeconds );
        fps = static_cast<float>( time.as(unit) );
    }

    if( fps <= 0.f )
    {
        fps = 24.0f;
    }

    return fps;
}

