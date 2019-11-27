#include <Bora.h>

using namespace std;
using namespace Bora;

static int winX = 0, winY = 800;
static AABB2f aabb( Vec2f(-1.f), Vec2f(1.f) );
static size_t Nx = 4, Ny = 6;
static Grid2D grid( Nx, Ny, aabb );

void Display()
{
    glClearColor( 0, 0, 0, 1 );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glEnable( GL_DEPTH_TEST );

	glViewport( 0, 0, winX, winY );

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
	gluOrtho2D( -0.5f, 0.5, -0.5, 0.5 );

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();

    glLineWidth(1);
    glColor(1.f);

    grid.glBeginNormalSpace();
    grid.draw();
    grid.glEndNormalSpace();

    glutSwapBuffers();
}

void GLUTKeyboardDown( unsigned char key, int x, int y )
{
    switch( key )
    {
        case 27: // esc
        {
            exit(0);
        }
    }
}

int main( int argc, char* argv[] )
{
	winX = winY * grid.nx() / (float)grid.ny();

    glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE );
	glutInitWindowSize( winX, winY );
    glutCreateWindow( "Grid2D Test" );
    glutDisplayFunc( Display );
    glutKeyboardFunc( GLUTKeyboardDown );

    glutMainLoop();

    return 0;
}

