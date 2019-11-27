//--------------------//
// DM_SimInfoRender.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2019.04.03                               //
//-------------------------------------------------------//

#pragma once

#include <HouCommon.h>

class DM_SimInfoRender : public DM_SceneRenderHook
{
    public:

        struct Data
        {
            std::string title;
            std::string message;

            Grid grid;

            int substeps;
            int mainParticles=0;
            int subParticles=0;
            int frames;

            DiffuseWaterMethod::Params diffParams;
            FLIPMethod::Params flipParams;

            Noise noise;
        };

        static std::map< size_t, DM_SimInfoRender::Data > map;

    public:

        DM_SimInfoRender( DM_VPortAgent &vport ) : DM_SceneRenderHook( vport, DM_VIEWPORT_ALL )
        {}

        void renderStringStream( RE_Render* r, const DM_SceneHookData &hook_data, std::stringstream& ss );

        virtual bool render( RE_Render* r, const DM_SceneHookData &hook_data )
        {
            std::stringstream ss;

            for( auto it=DM_SimInfoRender::map.begin(); it!=DM_SimInfoRender::map.end(); ++it )
            {
                const Data& data = it->second;

                if( data.mainParticles > 0 )
                { // Flip Solver Info
                    AABB3f box = data.grid.boundingBox();

                    ss<< data.title << std::endl;
                    ss<< " -Voxel Size : " << data.flipParams.voxelSize << std::endl;
                    ss<< " -Grid Res : " << data.grid.nx() << " X " << data.grid.ny() << " X " << data.grid.nz() << std::endl;
                    ss<< " -Grid Size : " << box.xWidth() << " X " << box.yWidth() << " X " << box.zWidth() << std::endl;
                    ss<< " -Sub Steps : " << data.flipParams.maxsubsteps << std::endl;
                    ss<< " -Particles  : " << data.mainParticles << std::endl;
                    ss<< " -Water Level : " << data.flipParams.wallLevel << std::endl;
                    ss<< " -Damping Width : " << data.flipParams.dampingBand << std::endl;
                    ss<< " -Adjacency Under : " << data.flipParams.adjacencyUnder << std::endl;
                    ss<< " -Projections : " << data.flipParams.projIteration << std::endl;
                    ss<< " -Redistances : " << data.flipParams.redistIteration << std::endl;
                    ss<< " -External Force : " << data.flipParams.extForce << std::endl;
                    ss<< std::endl;
                    ss<< data.message << std::endl;

                }
                else if( data.subParticles > 0 )
                { // Diffuse Sover Info

                }
            }

            renderStringStream( r, hook_data, ss );

            return false;
        }
};

class DM_SimInfoRenderHook : public DM_SceneHook
{
    public:

        DM_SimInfoRenderHook() : DM_SceneHook("Checkered Background", 0)
        {}

        virtual ~DM_SimInfoRenderHook()
        {}

        virtual DM_SceneRenderHook *newSceneRender(DM_VPortAgent &vport, DM_SceneHookType type, DM_SceneHookPolicy policy)
        {
            return new DM_SimInfoRender( vport );
        }

        virtual void retireSceneRender(DM_VPortAgent &vport, DM_SceneRenderHook *hook)
        {
            delete hook;
        }
};

