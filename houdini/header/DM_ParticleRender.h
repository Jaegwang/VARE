//---------------------//
// DM_ParticleRender.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2019.04.10                               //
//-------------------------------------------------------//

#pragma once

#include <HouCommon.h>

class DM_ParticleRender : public DM_SceneRenderHook
{
    private:

        static const char* vert_shader;
        static const char* frag_shader;

        RE_Shader* ptShader;

    public:

        struct Data
        {
            Particles* pts=0;

            int type=0;
            float adjacencyUnder;
            float curvatureOver;
            float height=-1e+20f;
            Vec3f color;
        };

        static std::map<size_t, DM_ParticleRender::Data > map;

    public:

        DM_ParticleRender( DM_VPortAgent &vport ) : DM_SceneRenderHook( vport, DM_VIEWPORT_ALL_3D ), ptShader( NULL )
	    {}

        virtual ~DM_ParticleRender() { delete ptShader; }

        virtual bool render( RE_Render* r, const DM_SceneHookData& hook_data );
};


class DM_ParticleRenderHook : public DM_SceneHook
{
    public:

        DM_ParticleRenderHook() : DM_SceneHook( "Bora Particles", 0 )
	    {}

        virtual DM_SceneRenderHook* newSceneRender( DM_VPortAgent &vport, DM_SceneHookType type, DM_SceneHookPolicy policy )
        {
            return new DM_ParticleRender( vport );
        }

        virtual void retireSceneRender( DM_VPortAgent &vport, DM_SceneRenderHook* hook )
        {
            delete hook;
        }
};

