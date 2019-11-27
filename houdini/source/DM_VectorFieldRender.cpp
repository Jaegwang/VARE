//--------------------------//
// DM_VectorFieldRender.cpp //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2019.04.03                               //
//-------------------------------------------------------//

#include <DM_VectorFieldRender.h>

const char* DM_VectorFieldRender::vert_shader = STRINGIFY(

    \n #version 150 \n

    uniform mat4 glH_ProjectMatrix;
    uniform mat4 glH_ViewMatrix;
    in vec3 P;
    in vec3 Cd;
    out vec4 clr;

    void main()
    {
        clr = vec4(Cd, 1.0);
        gl_Position = glH_ProjectMatrix * glH_ViewMatrix * vec4(P, 1.0);
    }

);
    
const char* DM_VectorFieldRender::frag_shader = STRINGIFY(

    \n #version 150 \n

    in vec4 clr;
    out vec4 color;

    void main()
    {
        color = clr;
    }
);

std::map< size_t, DM_VectorFieldRender::Data > DM_VectorFieldRender::map;

bool DM_VectorFieldRender::renderCurlWire( RE_Render *r, const DM_SceneHookData &hook_data, const DM_VectorFieldRender::Data& data )
{
    const AABB3f bb = data.grid.boundingBox();
    const Vec3f min = bb.minPoint();
    const Vec3f max = bb.maxPoint();

    const float lx = bb.xWidth();
    const float ly = bb.yWidth();
    const float lz = bb.zWidth();

    const size_t nx = data.grid.nx();
    const size_t ny = data.grid.ny();
    const size_t nz = data.grid.nz();

    const float dx = lx / (float)nx;
    const float dy = ly / (float)ny;
    const float dz = lz / (float)nz;

    const size_t N = (nx+ny+nz)*2;
    const size_t steps = 10;

    const float sub_dt = 1.f/(float)steps;

    const size_t total = N*steps*2;

    if( pos.size() < total )
    {
        pos = UT_Vector3FArray( total, total );
        col = UT_Vector3FArray( total, total );        
    }

    r->pushShader(myShader);

    {
        RE_Geometry geo( N*steps*2 );
        
        for( int n=0; n<N; ++n )
        {
            Vec3f p;

            if      ( data.axis == 0 ) p = Vec3f( data.offset*lx, Rand(n*5673)*ly, Rand(n*1265)*lz ) + min;
            else if ( data.axis == 1 ) p = Vec3f( Rand(n*5673)*lx, data.offset*ly, Rand(n*1265)*lz ) + min;
            else if ( data.axis == 2 ) p = Vec3f( Rand(n*1265)*lx, Rand(n*5673)*ly, data.offset*lz ) + min;

            float t = 0.f;

            for( int s=0; s<steps; s++ )
            {
                pos[n*steps*2 + s*2 + 0] = UT_Vector3( p.x, p.y, p.z );
                col[n*steps*2 + s*2 + 0] = UT_Vector3( t, 1.f-t, 0.f );

                p += data.noise.curl( p, data.dt*data.frame ) * sub_dt;
                t += sub_dt;

                pos[n*steps*2 + s*2 + 1] = UT_Vector3( p.x, p.y, p.z );
                col[n*steps*2 + s*2 + 1] = UT_Vector3( t, 1.f-t, 0.f );
            }
        }

        geo.createAttribute(r, "P" , RE_GPU_FLOAT32, 3, pos.array() );
        geo.createAttribute(r, "Cd", RE_GPU_FLOAT32, 3, col.array() );
        geo.connectAllPrims(r, 0, RE_PRIM_LINES);

        geo.draw(r, 0);
    }

    r->popShader();

    return false;
}

