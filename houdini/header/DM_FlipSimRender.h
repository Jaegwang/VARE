//------------------//
// DM_FlipSimRenderHook.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2019.04.03                               //
//-------------------------------------------------------//

#pragma once

#include <HouCommon.h>

class DM_FlipSimRender : public DM_SceneRenderHook
{
    private:

        static const char* vert_shader;
        static const char* frag_shader;

        RE_Shader* myShader;

    public:

        struct Data
        {
            Grid grid;
            FLIPMethod::Params params;
        };

        static std::map< size_t, DM_FlipSimRender::Data > map;

    public:

        DM_FlipSimRender( DM_VPortAgent &vport ) : DM_SceneRenderHook( vport, DM_VIEWPORT_ALL_3D ), myShader( NULL )
	    {}

        virtual ~DM_FlipSimRender() { delete myShader; }

        virtual bool render( RE_Render* r, const DM_SceneHookData& hook_data )
        {
            for( auto it=map.begin(); it!=map.end(); ++it )
            {
                const Data& data = it->second;

                renderGridWire( r, hook_data, data );
            }

            return false; // allow other hooks of this type to render
        }        

        bool renderGridWire( RE_Render* r, const DM_SceneHookData& hook_data, const DM_FlipSimRender::Data data );
};

class DM_FlipSimRenderHook : public DM_SceneHook
{
    public:

        DM_FlipSimRenderHook() : DM_SceneHook( "Flip Sim Render", 0 )
	    {}

        virtual DM_SceneRenderHook* newSceneRender( DM_VPortAgent &vport, DM_SceneHookType type, DM_SceneHookPolicy policy )
        {
            return new DM_FlipSimRender( vport );
        }

        virtual void retireSceneRender( DM_VPortAgent &vport, DM_SceneRenderHook* hook )
        {
            delete hook;
        }
};

