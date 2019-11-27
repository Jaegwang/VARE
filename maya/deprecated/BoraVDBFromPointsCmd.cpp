//--------------------------//
// BoraVDBFromPointsCmd.cpp //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2017.09.21                               //
//-------------------------------------------------------//

#include <BoraVDBFromPointsCmd.h>
#include <BoraVDBFromPoints.h>

#include <ri.h>
#include <openvdb/tools/LevelSetSphere.h>

const MString BoraVDBFromPointsCmd::name( "BoraVDBFromPointsCmd" );

MStatus BoraVDBFromPointsCmd::doIt( const MArgList& args )
{    
    MStatus stat = MS::kSuccess;

    MString nodeName;    
    args.get( 0, nodeName );

    BoraVDBFromPoints* data = BoraVDBFromPoints::instances[ nodeName.asChar() ];

    if( data == 0 )
    {
        return stat;
    }

    if( data->points )
    {
        Vec3fArray& points = *(data->points);

        if( points.num() > 0 )
        {
            MString workspace;
            MGlobal::executeCommand( MString("workspace -q -rd;"), workspace );
        
            int frame;
            MGlobal::executeCommand( MString("currentTime -q;"), frame );

            std::stringstream vdbFilePath;
            vdbFilePath << workspace.asChar() << "boraPointCloudVDB_" << frame << ".vdb";
       
            std::string name = "density";

            std::string dsoPath = std::getenv( "BORA_RMAN_DSO_PATH" );
            dsoPath.append("/BoraRmanVDBFromPoints.so");

            {
                std::stringstream ss;
                ss << "Procedural \"DynamicLoad\" [\"" << dsoPath << "\" ";
                ss << "\"" << "{";
                ss << "\\\"file\\\":" << "\\\"" << data->file << "\\\"" << ",";
                ss << "\\\"vdb\\\":" << "\\\"" << vdbFilePath.str() << "\\\"" << ",";
                ss << "\\\"name\\\":" << "\\\"" << name << "\\\"" << ",";            
                ss << "\\\"radius\\\":" << data->radius << ",";
                ss << "\\\"voxelSize\\\":" << data->voxelSize << ",";
                ss << "\\\"frame\\\":" << frame << "";
                ss << "}" << "\"] ";
                ss << "[-100000 100000 -100000 100000 -100000 100000]";

                RiArchiveRecord( RI_VERBATIM, (char*)ss.str().c_str() );                
            }
        }
    }

    if( data && data->velocities )
    {
        Vec3fArray& vel = *(data->velocities);        
    }

    return stat;
}

