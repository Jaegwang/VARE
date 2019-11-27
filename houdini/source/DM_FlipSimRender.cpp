//----------------------//
// DM_FlipSimRender.cpp //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2019.04.03                               //
//-------------------------------------------------------//

#include <DM_FlipSimRender.h>

const char* DM_FlipSimRender::vert_shader = STRINGIFY(

    \n #version 150 \n

    uniform mat4 glH_ProjectMatrix;
    uniform mat4 glH_ViewMatrix;
    in vec3 P;
    in vec3 Cd;
    out vec4 clr;

    void main()
    {
        clr = vec4(Cd, 0.5);
        gl_Position = glH_ProjectMatrix * glH_ViewMatrix * vec4(P, 1.0);
    }

);
    
const char* DM_FlipSimRender::frag_shader = STRINGIFY(

    \n #version 150 \n

    in vec4 clr;
    out vec4 color;

    void main()
    {
        color = clr;
    }
);

std::map< size_t, DM_FlipSimRender::Data > DM_FlipSimRender::map;

bool DM_FlipSimRender::renderGridWire( RE_Render *r, const DM_SceneHookData &hook_data, const DM_FlipSimRender::Data data )
{
    const Grid& grid = data.grid;

    if(!myShader)
    {
        myShader = RE_Shader::create("lines");
        myShader->addShader(r, RE_SHADER_VERTEX, vert_shader, "vertex", 0);
        myShader->addShader(r, RE_SHADER_FRAGMENT, frag_shader, "fragment", 0);
        myShader->linkShaders(r);
    }

    const AABB3f bb = grid.boundingBox();
    const Vec3f min = bb.minPoint();
    const Vec3f max = bb.maxPoint();

    const float x = bb.xWidth();
    const float y = bb.yWidth();
    const float z = bb.zWidth();

    const size_t nx = grid.nx();
    const size_t ny = grid.ny();
    const size_t nz = grid.nz();

    const float dx = x*0.2f;
    const float dy = y*0.2f;
    const float dz = z*0.2f;

    r->pushShader(myShader);

    // Draw grid box
    {
        RE_Geometry boxgeo(48);
        UT_Vector3FArray pos(48);
        UT_Vector3FArray col(48);

        pos[ 0] = UT_Vector3( min.x, min.y, min.z );
        pos[ 1] = pos[0] + UT_Vector3( dx, 0.f, 0.f );
        pos[ 2] = pos[0];
        pos[ 3] = pos[0] + UT_Vector3( 0.f, dy, 0.f );
        pos[ 4] = pos[0];
        pos[ 5] = pos[0] + UT_Vector3( 0.f, 0.f, dz );

        pos[ 6] = UT_Vector3( min.x+x, min.y, min.z );
        pos[ 7] = pos[6] + UT_Vector3( -dx, 0.f, 0.f );
        pos[ 8] = pos[6];
        pos[ 9] = pos[6] + UT_Vector3( 0.f, dy, 0.f );
        pos[10] = pos[6];
        pos[11] = pos[6] + UT_Vector3( 0.f, 0.f, dz );

        pos[12] = UT_Vector3( min.x+x, min.y+y, min.z );
        pos[13] = pos[12] + UT_Vector3( -dx, 0.f, 0.f );
        pos[14] = pos[12];
        pos[15] = pos[12] + UT_Vector3( 0.f, -dy, 0.f );
        pos[16] = pos[12];
        pos[17] = pos[12] + UT_Vector3( 0.f, 0.f, dz );

        pos[18] = UT_Vector3( min.x, min.y+y, min.z );
        pos[19] = pos[18] + UT_Vector3( dx, 0.f, 0.f );
        pos[20] = pos[18];
        pos[21] = pos[18] + UT_Vector3( 0.f, -dy, 0.f );
        pos[22] = pos[18];
        pos[23] = pos[18] + UT_Vector3( 0.f, 0.f, dz );

        pos[24] = UT_Vector3( min.x, min.y, min.z+z );
        pos[25] = pos[24] + UT_Vector3( dx, 0.f, 0.f );
        pos[26] = pos[24];
        pos[27] = pos[24] + UT_Vector3( 0.f, dy, 0.f );
        pos[28] = pos[24];
        pos[29] = pos[24] + UT_Vector3( 0.f, 0.f, -dz );

        pos[30] = UT_Vector3( min.x+x, min.y, min.z+z );
        pos[31] = pos[30] + UT_Vector3( -dx, 0.f, 0.f );
        pos[32] = pos[30];
        pos[33] = pos[30] + UT_Vector3( 0.f, dy, 0.f );
        pos[34] = pos[30];
        pos[35] = pos[30] + UT_Vector3( 0.f, 0.f, -dz );

        pos[36] = UT_Vector3( min.x+x, min.y+y, min.z+z );
        pos[37] = pos[36] + UT_Vector3( -dx, 0.f, 0.f );
        pos[38] = pos[36];
        pos[39] = pos[36] + UT_Vector3( 0.f, -dy, 0.f );
        pos[40] = pos[36];
        pos[41] = pos[36] + UT_Vector3( 0.f, 0.f, -dz );

        pos[42] = UT_Vector3( min.x, min.y+y, min.z+z );
        pos[43] = pos[42] + UT_Vector3( dx, 0.f, 0.f );
        pos[44] = pos[42];
        pos[45] = pos[42] + UT_Vector3( 0.f, -dy, 0.f );
        pos[46] = pos[42];
        pos[47] = pos[42] + UT_Vector3( 0.f, 0.f, -dz );

        for( int n=0; n<48; ++n ) col[n] = UT_Vector3( 0.8f, 0.8f, 0.f );

        boxgeo.createAttribute(r, "P" , RE_GPU_FLOAT32, 3, pos.array() );
        boxgeo.createAttribute(r, "Cd", RE_GPU_FLOAT32, 3, col.array() );
        boxgeo.connectAllPrims(r, 0, RE_PRIM_LINES);

        boxgeo.draw(r, 0);
    }

    if( data.params.enableWaterTank == true )
    {
        const float height = data.params.wallLevel;
        const float db = Max( data.params.dampingBand, data.params.voxelSize*0.5f );

        UT_Vector3FArray vertices(8);
        UT_Vector3FArray colors(8);

        vertices[0] = UT_Vector3( min.x, height, min.z );
        vertices[1] = vertices[0]+UT_Vector3( x  ,0.f ,0.f);
        vertices[2] = vertices[1]+UT_Vector3( 0.f,0.f ,z  );
        vertices[3] = vertices[2]+UT_Vector3(-x  ,0.f ,0.f);
        vertices[4] = vertices[0]+UT_Vector3( db, 0.f,  db);
        vertices[5] = vertices[1]+UT_Vector3(-db, 0.f,  db);
        vertices[6] = vertices[2]+UT_Vector3(-db, 0.f, -db);
        vertices[7] = vertices[3]+UT_Vector3( db, 0.f, -db);

        colors[0] = colors[1] = colors[2] = colors[3] = UT_Vector3( 0.f, 0.9f, 0.9f );
        colors[4] = colors[5] = colors[6] = colors[7] = UT_Vector3( 0.8f, 0.8f, 0.8f );

        RE_Geometry dampPlane(24);
        UT_Vector3FArray pos(24);
        UT_Vector3FArray col(24);

        pos[0] = vertices[0]; col[0] = colors[0];
        pos[1] = vertices[1]; col[1] = colors[1];
        pos[2] = vertices[4]; col[2] = colors[4];
        pos[3] = vertices[4]; col[3] = colors[4];
        pos[4] = vertices[1]; col[4] = colors[1];
        pos[5] = vertices[5]; col[5] = colors[5];

        pos[6]  = vertices[1]; col[6] = colors[1];
        pos[7]  = vertices[6]; col[7] = colors[6];
        pos[8]  = vertices[5]; col[8] = colors[5];
        pos[9]  = vertices[1]; col[9] = colors[1];
        pos[10] = vertices[2]; col[10] = colors[2];
        pos[11] = vertices[6]; col[11] = colors[6];

        pos[12] = vertices[6]; col[12] = colors[6];
        pos[13] = vertices[2]; col[13] = colors[2];
        pos[14] = vertices[3]; col[14] = colors[3];
        pos[15] = vertices[7]; col[15] = colors[7];
        pos[16] = vertices[6]; col[16] = colors[6];
        pos[17] = vertices[3]; col[17] = colors[3];

        pos[18] = vertices[0]; col[18] = colors[0];
        pos[19] = vertices[7]; col[19] = colors[7];
        pos[20] = vertices[3]; col[20] = colors[3];
        pos[21] = vertices[0]; col[21] = colors[0];
        pos[22] = vertices[4]; col[22] = colors[4];
        pos[23] = vertices[7]; col[23] = colors[7];

        dampPlane.createAttribute(r, "P" , RE_GPU_FLOAT32, 3, pos.array() );
        dampPlane.createAttribute(r, "Cd", RE_GPU_FLOAT32, 3, col.array() );
        dampPlane.connectAllPrims(r, 0, RE_PRIM_TRIANGLES);

        dampPlane.draw(r,0);
    }
   
    {
    }

    r->popShader();

    return false;
}

