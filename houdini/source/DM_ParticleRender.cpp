//-----------------------//
// DM_ParticleRender.cpp //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2019.04.10                               //
//-------------------------------------------------------//

#include <DM_ParticleRender.h>

const char* DM_ParticleRender::vert_shader = STRINGIFY(

    \n #version 150 \n

    uniform mat4 glH_ProjectMatrix;
    uniform mat4 glH_ViewMatrix;

    uniform int type;
    uniform float height;
    uniform float alpha;
    uniform vec3 uColor;

    in vec3 P;
    in float attrib;

    out vec4 clr;

    void main()
    {
        float a = 0.0;

        if( type == 0 ) // adjacency
        {
            a = attrib - alpha;
            if( a < 0.0 ) a = -a/alpha;
            else a = 0.0;
        }
        else if( type == 1 ) // curvature
        {
            a = (attrib-alpha)/(3.f-alpha);
            if( a < 0.0 ) a = 0.0;
            else if( a > 1.0 ) a = 1.0;
        }

        clr = vec4(uColor*a, 1.0);

        gl_Position = glH_ProjectMatrix * glH_ViewMatrix * vec4(P, 1.0);
    }
);

const char* DM_ParticleRender::frag_shader = STRINGIFY(

    \n #version 150 \n

    in vec4 clr;
    out vec4 color;

    void main()
    {
        color = clr;
    }
);

std::map<size_t, DM_ParticleRender::Data > DM_ParticleRender::map;

bool DM_ParticleRender::render( RE_Render* r, const DM_SceneHookData& hook_data )
{
    if(!ptShader)
    {
        ptShader = RE_Shader::create( "particles" );
        ptShader->addShader( r, RE_SHADER_VERTEX, vert_shader, "vertex", 0 );
        ptShader->addShader( r, RE_SHADER_FRAGMENT, frag_shader, "fragment", 0 );
        ptShader->linkShaders( r );
    }

    for( auto it=DM_ParticleRender::map.begin(); it!=DM_ParticleRender::map.end(); ++it )
    {
        if( it->second.pts == 0 ) continue;

        const DM_ParticleRender::Data& data = it->second;
        const Vec3f& c = data.color;

        ptShader->bindInt( r, "type", data.type );
        ptShader->bindVector( r, "uColor", UT_Vector3F( c.x, c.y, c.z ) );
        ptShader->bindFloat( r, "height", data.height );

        if( data.type == 0 ) ptShader->bindFloat( r, "alpha", data.adjacencyUnder );
        if( data.type == 1 ) ptShader->bindFloat( r, "alpha", data.curvatureOver  );

        r->pushShader( ptShader );
        r->pushPointSize( 5.0 );

        size_t s = data.pts->position.size();

        Vec3f* pos = data.pts->position.pointer();
        float* adj = data.pts->adjacency.pointer();
        float* cur = data.pts->curvature.pointer();

        RE_Geometry geo( s );
        
        geo.createAttribute( r, "P", RE_GPU_FLOAT32, 3, pos );

        if( data.type == 0 ) geo.createAttribute( r, "attrib", RE_GPU_FLOAT32, 1, adj );
        if( data.type == 1 ) geo.createAttribute( r, "attrib", RE_GPU_FLOAT32, 1, cur );

        geo.connectAllPrims( r, 0, RE_PRIM_POINTS );

        geo.draw( r, 0 );

        r->popPointSize();
        r->popShader();        
    }

    return false;
}

