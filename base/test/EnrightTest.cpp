//----------------//
// EnrightTest.cpp //
//-------------------------------------------------------//
// author: Julie Jang @ Dexter Studios                   //
// last update: 2018.04.10                               //
//-------------------------------------------------------//

#include <Bora.h>
#include <GL/glut.h>

using namespace Bora;

float _cfl_number = 1.f;
static int _minSubSteps = 1;
static int _maxSubSteps = 2;
static AdvectionScheme _advScheme = kRK3;
static float _T = 3.f, _Dt = 1.f/24.f;
static int winX = 0, winY = 800;
static int winID;
float 		   	 _t;
int				 _gridRes = 200;
EnrightTest		 _enrightTest;
bool			 _isDisplaySphere = false;
ScalarDenseField _sphereLVS;


void drawScalarField()
{
	glPointSize( (winX/_gridRes)+1.f );
	if( _isDisplaySphere )
	{
		int n = _gridRes;
		float minValue = _sphereLVS.minValue();
		AABB3f aabb = _sphereLVS.boundingBox();
		float L = aabb.width(0);
		Vec3f minPt = aabb.minPoint();
	
		glPointSize( 5 );
		glBegin( GL_POINTS );
		for( int k=0; k<n; k++ )
		for( int j=0; j<n; j++ )
		for( int i=0; i<n; i++ )
		{
			float value = _sphereLVS( i, j, k );
			if( value<EPSILON )
			{
				float normalizedValue = value/minValue;
				glColor( normalizedValue, 0.f, 0.f );
				Vec3f p = _sphereLVS.worldPoint( Vec3f(i, j, k) ) + Vec3f(L/(float)n/2.f);
				glVertex( p.x, p.y, p.z );
			}
		}
		glEnd();
	}
	else { _enrightTest.drawScalarField(); }
	glPointSize( 10.f );
	glColor( 0.f, 1.f, 0.f );
	glVertex3f( 0.f, 0.f, 0.f );
	glColor( 0.f, 0.f, 1.f );
	glVertex3f( 1.f, 1.f, 1.f );
	glEnd();

}

void Display()
{
    glClearColor( 0.4f, 0.4f, 0.4f, 1.f );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glEnable( GL_DEPTH_TEST );
	glViewport ( 0, 0, winX, winY );
	glMatrixMode ( GL_PROJECTION );
	glLoadIdentity ();
	gluPerspective( 90.f, 1.0, 1.f, 101.0 );
	//gluPerspective( 40.f, 1.0, 0.5, 7.0 );
	glMatrixMode ( GL_MODELVIEW );
	glLoadIdentity ();
	gluLookAt ( 0.0, 0.0, 50.f, 0.f, 0.f, 0.f, 0.0, 1.f, 0.0 );
	//gluLookAt ( 0.0, 0.0, 3.f, 0.5f, 0.5f, 0.5f, 0.0, 1.f, 0.0 );
	glClearColor ( 0.4f, 0.4f, 0.4f, 1.0f );
	glClear ( GL_COLOR_BUFFER_BIT );

    drawScalarField();

    glutSwapBuffers();
}

static void Idle ( void )
{
	if( (_t>=_Dt) && (_t<(_T-_Dt)) ) { _t +=_Dt; int substeps = _enrightTest.update(_t); COUT<<"substeps: "<<substeps<<ENDL; }
	glutSetWindow ( winID );
	glutPostRedisplay ();
}

void GLUTKeyboard( unsigned char key, int x, int y )
{
    switch( key )
    {
        case 27: // esc
        {
            exit(0);
			break;
        }
        case 32: // space
        {
            _t+=_Dt;
			_enrightTest.update(_t);
			break;
        }
		case 49: // 1
		{
			_isDisplaySphere = true;
			break;
		}
		case 50: // 2
		{
			_isDisplaySphere = false;
			break;
		}
    }
}


void GLUTSpecialKeyboard( int key, int x, int y )
{
    switch( key )
    {
		case GLUT_KEY_UP:
		{
			_cfl_number += 0.1;
			_enrightTest.setAdvectionConditions( _cfl_number, _minSubSteps, _maxSubSteps, _Dt, _advScheme );
			COUT<<"cfl_number: "<<_cfl_number<<ENDL;
			break;
		}
		case GLUT_KEY_DOWN:
		{
			_cfl_number -= 0.1;
			_enrightTest.setAdvectionConditions( _cfl_number, _minSubSteps, _maxSubSteps, _Dt, _advScheme );
			COUT<<"cfl_number: "<<_cfl_number<<ENDL;
			break;
		}
    }
}


int main( int argc, char** argv )
{
	winX = winY;
	_enrightTest.initialize( _gridRes );
	_enrightTest.setAdvectionConditions( _cfl_number, _minSubSteps, _maxSubSteps, _Dt, _advScheme );

	AABB3f aabb = AABB3f( Vec3f(-50.f), Vec3f(50.f) );
	Grid grd = Grid( _gridRes, aabb );
	_sphereLVS.initialize( grd, kUnified );
	const float Lx = aabb.width(0);
	const Vec3f minPt = aabb.minPoint();
	Vec3f center = Vec3f(0.35*Lx) + minPt;
	float radius = 0.15*Lx;
	SetSphere( center, radius, _sphereLVS );


	glutInit( &argc, argv );
	glutInitDisplayMode ( GLUT_RGBA | GLUT_DOUBLE );
	glutInitWindowSize ( winX, winY );
	winID = glutCreateWindow ( "Enright Test" );
    glutDisplayFunc( Display );
    glutIdleFunc( Idle );
    glutKeyboardFunc( GLUTKeyboard );
	glutSpecialFunc( GLUTSpecialKeyboard );

	glutMainLoop();

	return 0;
}

