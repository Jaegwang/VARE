//-----------------//
// GLDrawField.cpp //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.01.26                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

bool GLDrawField::_initial = false;

GLDrawField::GLDrawField()
{
    if( _initial == true ) return;

    initialize(); 
    _initial = true;
}

void GLDrawField::drawGridBox( const Grid& grid )
{
    AABB3f bb = grid.boundingBox();

    const Vec3f& min = bb.minPoint();
    const Vec3f& max = bb.maxPoint();

    glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );

    glBegin(GL_QUADS);

    glVertex3f(max.x, max.y, min.z);
    glVertex3f(min.x, max.y, min.z);
    glVertex3f(min.x, max.y, max.z);
    glVertex3f(max.x, max.y, max.z);

    glVertex3f(max.x, min.y, max.z);
    glVertex3f(min.x, min.y, max.z);
    glVertex3f(min.x, min.y, min.z);
    glVertex3f(max.x, min.y, min.z);

    glVertex3f(max.x, max.y, max.z);
    glVertex3f(min.x, max.y, max.z);
    glVertex3f(min.x, min.y, max.z);
    glVertex3f(max.x, min.y, max.z);

    glVertex3f(max.x, min.y, min.z);
    glVertex3f(min.x, min.y, min.z);
    glVertex3f(min.x, max.y, min.z);
    glVertex3f(max.x, max.y, min.z);

    glVertex3f(min.x, max.y, max.z);
    glVertex3f(min.x, max.y, min.z);
    glVertex3f(min.x, min.y, min.z);
    glVertex3f(min.x, min.y, max.z);

    glVertex3f(max.x, max.y, min.z);
    glVertex3f(max.x, max.y, max.z);
    glVertex3f(max.x, min.y, max.z);
    glVertex3f(max.x, min.y, min.z);
    
    glEnd();

    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
}

void GLDrawField::drawVelocityField( const VectorDenseField& field, const float dt )
{
    const size_t N = field.numVoxels();

    glBegin( GL_LINES );
    for( int n=0; n<N; ++n )
    {
        size_t i,j,k;
        field.cellIndices( n, i, j, k );

        Vec3f v = field[n];
        Vec3f c = field.cellCenter( i,j,k );

        Vec3f tar = c + v*dt;

        glVertex3f( c.x, c.y, c.z );
        glVertex3f( tar.x, tar.y, tar.z );
    }
    glEnd();
}

void GLDrawField::drawDensityField( const ScalarDenseField& field )
{
    const size_t N = field.numVoxels();

    glBegin( GL_POINTS );
    for( int n=0; n<N; ++n )
    {
        if( field[n] < 1e-05f ) continue;

        size_t i,j,k;
        field.cellIndices( n, i, j, k );

        Vec3f c = field.cellCenter( i,j,k );

        glVertex3f( c.x, c.y, c.z );
    }
    glEnd();    
}

void GLDrawField::initialize()
{
    
}

BORA_NAMESPACE_END

