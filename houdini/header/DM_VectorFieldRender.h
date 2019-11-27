//------------------------//
// DM_VectorFieldRender.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2019.04.03                               //
//-------------------------------------------------------//

#pragma once

#include <HouCommon.h>

class DM_VectorFieldRender : public DM_SceneRenderHook
{
    private:

        static const char* vert_shader;
        static const char* frag_shader;
        RE_Shader *myShader;

        UT_Vector3FArray pos;
        UT_Vector3FArray col;

    public:

        struct Data
        {
            Grid  grid;
            Noise noise;
            int   axis=0;
            float offset=0.5f;
            float frame=0;
            float dt=1.f/24.f;
        };

        static std::map< size_t, DM_VectorFieldRender::Data > map;

    public:

        DM_VectorFieldRender( DM_VPortAgent &vport ) : DM_SceneRenderHook( vport, DM_VIEWPORT_ALL_3D ), myShader( NULL )
	    {}

        virtual ~DM_VectorFieldRender() { delete myShader; }

        virtual bool render( RE_Render *r, const DM_SceneHookData &hook_data )
        {
            if(!myShader)
            {
                myShader = RE_Shader::create("lines");
                myShader->addShader(r, RE_SHADER_VERTEX, vert_shader, "vertex", 0);
                myShader->addShader(r, RE_SHADER_FRAGMENT, frag_shader, "fragment", 0);
                myShader->linkShaders(r);
            }    

            for( auto it=DM_VectorFieldRender::map.begin(); it!=DM_VectorFieldRender::map.end(); ++it )
            {
                DM_VectorFieldRender::Data& data = it->second;
                renderCurlWire( r, hook_data, data );
            }

            return false; // allow other hooks of this type to render
        }

        bool renderCurlWire( RE_Render *r, const DM_SceneHookData &hook_data, const DM_VectorFieldRender::Data& data );
};

class DM_VectorFieldRenderHook : public DM_SceneHook
{
    public:

        DM_VectorFieldRenderHook() : DM_SceneHook( "Field", 0 )
	    {}

        virtual DM_SceneRenderHook* newSceneRender( DM_VPortAgent &vport, DM_SceneHookType type, DM_SceneHookPolicy policy )
        {
            return new DM_VectorFieldRender( vport );
        }

        virtual void retireSceneRender( DM_VPortAgent &vport, DM_SceneRenderHook* hook )
        {
            delete hook;
        }
};

