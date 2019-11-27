//----------//
// Test.cpp //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.02.22                               //
//-------------------------------------------------------//

#include <Bora.h>

using namespace std;
using namespace Bora;

void Display( int width, int height )
{
	glViewport( 0, 0, width, height );

	glClearColor( 0.f, 0.f, 0.f, 0.f );

	glBegin( GL_TRIANGLES );
	{
		glColor3f( 1.f, 0.f, 0.f ); glVertex2f( -1.f, -1.f );
		glColor3f( 0.f, 1.f, 0.f ); glVertex2f(  0.f, +1.f );
		glColor3f( 0.f, 0.f, 1.f ); glVertex2f( +1.f, -1.f );
	}
	glEnd();

	glutSwapBuffers();
}

void InitGL()
{
	GLenum status = glewInit();

	if( status != GLEW_OK )
	{
		cout << glewGetErrorString(status) << endl;
		exit(0);
	}

	cout << "OpenGL version: " << glGetString(GL_VERSION) << " supported." << endl;
	cout << "GLEW version: " << glewGetString(GLEW_VERSION) << endl;
}

int main( int argc, char** argv )
{
    // Can you do offscreen rendering without creating window?
    glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE );
    glutCreateWindow( "OpenGL Test" );

    InitGL();

    GLCapture capture;

    capture.initialize( 2000, 2000 );

    capture.begin();
    {
        Display( capture.width(), capture.height() );
    }
    capture.end();

    capture.save( "result.exr" );

	return 0;
}

