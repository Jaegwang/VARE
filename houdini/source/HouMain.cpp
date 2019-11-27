//-----------------//
// HoudiniMain.cpp //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2019.04.10                               //
//-------------------------------------------------------//

#include <UT/UT_DSOVersion.h>

#include <SOP_ClusterPoints.h>
#include <SOP_FlipWater.h>
#include <SOP_DiffuseWater.h>
#include <SOP_Unknown.h>
#include <SOP_Distribute.h>
#include <SOP_ParticleView.h>
#include <SOP_CurlNoiseField.h>
#include <SOP_WrappingPoints.h>
#include <SOP_BulletBasic.h>

#include <DOP_ChainForce.h>

#include <VEX_OceanSampler.h>

#include <SIM_Unknown.h>

#include <ROP_Tractor.h>

// for OGL Render
#include <DM_MessageRender.h>
#include <DM_SimInfoRender.h>
#include <DM_FlipSimRender.h>
#include <DM_VectorFieldRender.h>
#include <DM_ParticleRender.h>

// SOPs
void newSopOperator( OP_OperatorTable *table )
{
    REGISTER_SOP( table, SOP_ClusterPoints, 1, 1, 0 );

    REGISTER_SOP( table, SOP_FlipWater, 1, 4, "DOP_flipsolver" );
    
    REGISTER_SOP( table, SOP_DiffuseWater, 1, 4, "SHELF_whitecaps" );

    REGISTER_SOP( table, SOP_Distribute, 1, 1, 0 );

    REGISTER_SOP( table, SOP_Unknown, 1, 2, 0 );

    REGISTER_SOP( table, SOP_ParticleView, 1, 1, 0 );

    REGISTER_SOP( table, SOP_CurlNoiseField, 1, 1, "DOP_windforce" );

    REGISTER_SOP( table, SOP_WrappingPoints, 1, 1, 0 );

    REGISTER_SOP( table, SOP_BulletBasic, 1, 1, 0 );
}

// DOPs
void newDopOperator( OP_OperatorTable *table )
{
    REGISTER_DOP( table, DOP_ChainForce, 1, 1, 0 );
}

// VEXs
void newVEXOp( void* )
{
    REGISTER_VEX( VEX_OceanSampler );
}

// SIMs
void initializeSIM( void* )
{
    IMPLEMENT_DATAFACTORY( SIM_Unknown );
}

// ROPs
void newDriverOperator( OP_OperatorTable *table )
{
    REGISTER_ROP( table, ROP_Tractor, 0, 9999 );
}

// DSO hook registration function
void
newRenderHook(DM_RenderTable *table)
{
    table->registerSceneHook( new DM_MessageRenderHook, DM_HOOK_FOREGROUND, DM_HOOK_AFTER_NATIVE );
    table->installSceneOption( "bora_message", "Bora Message" );

    table->registerSceneHook( new DM_SimInfoRenderHook, DM_HOOK_FOREGROUND, DM_HOOK_AFTER_NATIVE );
    table->installSceneOption( "bora_info", "Bora Info" );

    table->registerSceneHook( new DM_FlipSimRenderHook, DM_HOOK_UNLIT, DM_HOOK_AFTER_NATIVE );
    table->installSceneOption( "bora_flip", "Bora Flip" );

    table->registerSceneHook( new DM_VectorFieldRenderHook, DM_HOOK_UNLIT, DM_HOOK_AFTER_NATIVE );
    table->installSceneOption( "bora_vectorfield", "Bora Vector Field" );

    table->registerSceneHook( new DM_ParticleRenderHook, DM_HOOK_UNLIT, DM_HOOK_AFTER_NATIVE );
    table->installSceneOption( "bora_particles", "Bora Particles" );
}

