//--------------------//
// DM_MessageRender.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2019.04.03                               //
//-------------------------------------------------------//

#pragma once

#include <HouCommon.h>

class DM_MessageRender : public DM_SceneRenderHook
{
    public:

        struct Data
        {
            std::stringstream message;
            float time=3.f;
        };

        static std::map< size_t, DM_MessageRender::Data > map;

    public:

        DM_MessageRender( DM_VPortAgent &vport ) : DM_SceneRenderHook( vport, DM_VIEWPORT_ALL )
        {}

        void renderStringStream( RE_Render* r, const DM_SceneHookData &hook_data, std::stringstream& ss );

        virtual bool render( RE_Render* r, const DM_SceneHookData &hook_data )
        {
            if( r == 0 ) return false;

            for( auto it=DM_MessageRender::map.begin(); it!=DM_MessageRender::map.end(); ++it )
            {
                DM_MessageRender::Data& data = it->second;

                renderStringStream( r, hook_data, data.message );

                data.time -= 1.f/30.f;
            }

            return false;
        }
};

class DM_MessageRenderHook : public DM_SceneHook
{
    public:

        DM_MessageRenderHook() : DM_SceneHook("Checkered Background", 0)
        {}

        virtual ~DM_MessageRenderHook()
        {}

        virtual DM_SceneRenderHook *newSceneRender(DM_VPortAgent &vport, DM_SceneHookType type, DM_SceneHookPolicy policy)
        {
            return new DM_MessageRender( vport );
        }

        virtual void retireSceneRender(DM_VPortAgent &vport, DM_SceneRenderHook *hook)
        {
            delete hook;
        }
};

