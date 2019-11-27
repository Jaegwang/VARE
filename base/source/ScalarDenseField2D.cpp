//------------------------//
// ScalarDenseField2D.cpp //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
//         Julie Jang @ Dexter Studios                   //
// last update: 2018.04.25                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

ScalarDenseField2D::ScalarDenseField2D()
{
    // nothing to do
}

float ScalarDenseField2D::min() const
{
//    return DenseField2D<float>::minValue();
    return 0.f;
}

float ScalarDenseField2D::max() const
{
//    return DenseField2D<float>::maxValue();
    return 0.f;
}

void ScalarDenseField2D::drawLevelset( float maxInsideDistance, float maxOutsideDistance ) const
{
//    glClearColor( 0, 0, 0, 1 );
//    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
//    glEnable( GL_DEPTH_TEST );
//
//	glViewport( 0, 0, winX, winY );

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
	gluOrtho2D( -0.5f, 0.5, -0.5, 0.5 );

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();

    glLineWidth(1);
    glColor(1.f);

    Grid2D::glBeginNormalSpace();

    const size_t iMax = Grid2D::nx();
    const size_t jMax = Grid2D::ny();

    Vec3f c0, c1, c2, c3;
    Vec2f p0, p1, p2, p3;

    glBegin( GL_QUADS );
    {
        for( size_t j=0; j<jMax; ++j )
        for( size_t i=0; i<iMax; ++i )
        {
            const float& v0 = DenseField2D<float>::operator()( i  , j   );
            const float& v1 = DenseField2D<float>::operator()( i+1, j   );
            const float& v2 = DenseField2D<float>::operator()( i+1, j+1 );
            const float& v3 = DenseField2D<float>::operator()( i  , j+1 );

            if( v0 < 0.f ) { c0.set( 1.f, 1.f - -v0/maxInsideDistance, 0.f ); }
            else           { c0.set( 0.f, v0/maxInsideDistance,        1.f ); }

            if( v1 < 0.f ) { c1.set( 1.f, 1.f - -v1/maxInsideDistance, 0.f ); }
            else           { c1.set( 0.f, v1/maxInsideDistance,        1.f ); }

            if( v2 < 0.f ) { c2.set( 1.f, 1.f - -v2/maxInsideDistance, 0.f ); }
            else           { c2.set( 0.f, v2/maxInsideDistance,        1.f ); }

            if( v3 < 0.f ) { c3.set( 1.f, 1.f - -v3/maxInsideDistance, 0.f ); }
            else           { c3.set( 0.f, v3/maxInsideDistance,        1.f ); }

            p0.set( i  , j   );
            p1.set( i+1, j   );
            p2.set( i+1, j+1 );
            p3.set( i  , j+1 );

            glColor( c0 );   glVertex( p0 );
            glColor( c1 );   glVertex( p1 );
            glColor( c2 );   glVertex( p2 );
            glColor( c3 );   glVertex( p3 );
        }
    }
    glEnd();

    Grid2D::glEndNormalSpace();

//    glutSwapBuffers();
}

void ScalarDenseField2D::drawSmoke() const
{
    const size_t iMax = Grid2D::nx() - 1;
    const size_t jMax = Grid2D::ny() - 1;

    Vec2f p0, p1, p2, p3;

    glBegin( GL_QUADS );
    {
        for( size_t j=0; j<jMax; ++j )
        for( size_t i=0; i<iMax; ++i )
        {
            const float& v0 = Clamp( DenseField2D<float>::operator()( i  , j   ), 0.f, 1.f );
            const float& v1 = Clamp( DenseField2D<float>::operator()( i+1, j   ), 0.f, 1.f );
            const float& v2 = Clamp( DenseField2D<float>::operator()( i+1, j+1 ), 0.f, 1.f );
            const float& v3 = Clamp( DenseField2D<float>::operator()( i  , j+1 ), 0.f, 1.f );

            p0.set( i  , j   );
            p1.set( i+1, j   );
            p2.set( i+1, j+1 );
            p3.set( i  , j+1 );

            glColor( v0 );   glVertex( p0 );
            glColor( v1 );   glVertex( p1 );
            glColor( v2 );   glVertex( p2 );
            glColor( v3 );   glVertex( p3 );
        }
    }
    glEnd();
}

BORA_NAMESPACE_END

