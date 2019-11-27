//-------------//
// HouCommon.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2019.04.03                               //
//-------------------------------------------------------//

#pragma once

#include <Bora.h>

#include <GU/GU_Detail.h>
#include <GU/GU_PrimPoly.h>
#include <GU/GU_PrimVDB.h>
#include <GA/GA_Handle.h>

#include <OP/OP_AutoLockInputs.h>
#include <OP/OP_Operator.h>
#include <OP/OP_OperatorTable.h>
#include <OP/OP_Network.h>
#include <OP/OP_Director.h>

#include <PRM/PRM_Include.h>

// GEO
#include <GEO/GEO_PrimPart.h>
#include <GEO/GEO_PrimVDB.h>

// SOP
#include <SOP/SOP_Node.h>

// ROP
#include <ROP/ROP_Node.h>
#include <ROP/ROP_Templates.h>

// SIM
#include <SIM/SIM_DopDescription.h>
#include <SIM/SIM_GeometryCopy.h>
#include <SIM/SIM_DataFilter.h>
#include <SIM/SIM_Object.h>
#include <SIM/SIM_ObjectArray.h>
#include <SIM/SIM_Engine.h>
#include <SIM/SIM_Force.h>
#include <SIM/SIM_DataFilter.h>
#include <SIM/SIM_Relationship.h>
#include <SIM/SIM_RelationshipGroup.h>
#include <SIM/SIM_SingleSolver.h>
#include <SIM/SIM_OptionsUser.h>
#include <SIM/SIM_Utils.h>
#include <SIM/SIM_Force.h>

// GAS
#include <GAS/GAS_SubSolver.h>
#include <GAS/GAS_Utils.h>

// DOP
#include <DOP/DOP_Node.h>
#include <DOP/DOP_PRMShared.h>
#include <DOP/DOP_InOutInfo.h>
#include <DOP/DOP_Operator.h>
#include <DOP/DOP_Engine.h>

// UT
#include <UT/UT_JSONParser.h>

#include <DM/DM_RenderTable.h>
#include <DM/DM_SceneHook.h>
#include <DM/DM_VPortAgent.h>
#include <DM/DM_GeoDetail.h>

#include <RE/RE_Font.h>
#include <RE/RE_Geometry.h>
#include <RE/RE_Render.h>
#include <RE/RE_Shader.h>
#include <RE/RE_Texture.h>
#include <RE/RE_VertexArray.h>
#include <GUI/GUI_DisplayOption.h>
#include <GUI/GUI_ViewParameter.h>
#include <GUI/GUI_ViewState.h>

// VEX
#include <VEX/VEX_VexOp.h>

// OpenVDB 4.0
#include <openvdb/openvdb.h>
#include <openvdb/tools/ChangeBackground.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/VolumeToMesh.h>

#include <HouTools.h>
#include <HouRegisters.h>

#define STRINGIFY(x) #x

using namespace Bora;

