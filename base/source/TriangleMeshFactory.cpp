//-------------------------//
// TriangleMeshFactory.cpp //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.01.16                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

void CreatePlane( size_t Nx, size_t Ny, float Lx, float Ly, Axis axis, TriangleMesh* o_meshPtr )
{
    if( !o_meshPtr ) { return; }

    const float Dx = Lx / (float)Nx;
    const float Dy = Ly / (float)Ny;

    TriangleMesh& mesh = *o_meshPtr;
    mesh.clear();

    // vertex positions
    Vec3fArray& vPos = mesh.pos;
    {
        size_t index = 0;

        vPos.resize( (Nx+1)*(Ny+1) );

        for( size_t j=0; j<=Ny; ++j )
        {
            for( size_t i=0; i<=Nx; ++i )
            {
                Vec3f& p = vPos[index++];

                switch( axis )
                {
                    case Axis::kXaxis:
                    {
                        p.y = -0.5f*Lx + ( i * Dx );
                        p.z = -0.5f*Ly + ( j * Dy );
                        break;
                    }

                    case Axis::kYaxis:
                    {
                        p.z = -0.5f*Lx + ( i * Dx );
                        p.x = -0.5f*Ly + ( j * Dy );
                        break;
                    }

                    case Axis::kZaxis:
                    {
                        p.x = -0.5f*Lx + ( i * Dx );
                        p.y = -0.5f*Ly + ( j * Dy );
                        break;
                    }
                }
            }
        }
    }

    // triangle indices
    IndexArray& indices = mesh.indices;
    {
        size_t index = 0;

        indices.resize( 2*3*Nx*Ny );

        for( size_t j=0; j<Ny; ++j )
        {
            for( size_t i=0; i<Nx; ++i )
            {
                // the first triangle
                indices[index++] = Wrap(int(i+1),int(Nx+1)) + (Nx+1) * Wrap(int(j  ),int(Ny+1));
                indices[index++] = Wrap(int(i  ),int(Nx+1)) + (Nx+1) * Wrap(int(j  ),int(Ny+1));
                indices[index++] = Wrap(int(i+1),int(Nx+1)) + (Nx+1) * Wrap(int(j+1),int(Ny+1));

                // the second triangle
                indices[index++] = Wrap(int(i  ),int(Nx+1)) + (Nx+1) * Wrap(int(j  ),int(Ny+1));
                indices[index++] = Wrap(int(i  ),int(Nx+1)) + (Nx+1) * Wrap(int(j+1),int(Ny+1));
                indices[index++] = Wrap(int(i+1),int(Nx+1)) + (Nx+1) * Wrap(int(j+1),int(Ny+1));
            }
        }
    }
}

BORA_NAMESPACE_END

