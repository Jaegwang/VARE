#include <Bora.h>

using namespace std;
using namespace Bora;

static float _cfl_number = 1.f;
static int _minSubSteps = 1;
static int _maxSubSteps = 100;
static AdvectionScheme _advScheme = kRK3;
static float _T = 8.f, _Dt = 1.f/24.f;
static int winX = 0, winY = 800;
static int winID;
float 		   	   _t;
int				   _gridRes = 300;
EnrightTest2D	   _enrightTest2D;
bool			   _isDisplayCircle = false;
ScalarDenseField2D _circleLVS;


void Display()
{
    glClearColor( 0.4f, 0.4f, 0.4f, 1.f );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glEnable( GL_DEPTH_TEST );
	glViewport( 0, 0, winX, winY );
//    glMatrixMode( GL_PROJECTION );
//    glLoadIdentity();
//	//gluOrtho2D( 0.0, 1.0, 0.0, 1.0 );
//	gluOrtho2D( -0.5, 0.5, -0.5, 0.5 );
//    glMatrixMode( GL_MODELVIEW );
//    glLoadIdentity();

//  grid.glBeginNormalSpace();
//	glColor3f ( 1.0f, 1.0f, 1.0f );
//  grid.draw();
//  grid.glEndNormalSpace();

	//_enrightTest2D.drawVelocityField();


//	float minValue = _circleLVS.minValue();
//	AABB2f aabb = _circleLVS.boundingBox();
//	float L = aabb.width(0);
//	Vec2f minPt = aabb.minPoint();
//	glPointSize( (winX/(float)_gridRes)+1.f );
//	glBegin( GL_POINTS );
//	for( int j=0; j<_gridRes; j++ )
//	for( int i=0; i<_gridRes; i++ )
//	{
//		const float& value = _circleLVS( i, j );
//
//		if( value<EPSILON )
//		{
//			float normalizedValue = value/minValue;
//			glColor( normalizedValue, 0.f, 0.f );
//			//glColor( value, 0.f, 0.f );
//			Vec2f p = (_circleLVS.worldPoint( Vec2f(i,j)+Vec2f(0.5f) )-minPt)/L-Vec2f(0.5f);
//			glVertex2f( p.x, p.y );
//		}
//	}
//	glPointSize( 10.f );
//	glColor( 0.f, 0.f, 1.f );
//	Vec2f c = (_circleLVS.worldPoint( Vec2f(_gridRes*.5)+Vec2f(0.5f) )-minPt)/L-Vec2f(0.5f);
//	glVertex2f( c.x, c.y );
//	glEnd();

	if( _isDisplayCircle )
	{
		AABB2f aabb = _circleLVS.boundingBox();
		float diagonalLength = aabb.diagonalLength();
		float minValue = _circleLVS.minValue();
		_circleLVS.drawLevelset( diagonalLength, diagonalLength );
	}
	else { _enrightTest2D.drawScalarField(); }

    glutSwapBuffers();
}


static void Idle ( void )
{
	if( (_t>=_Dt) && (_t<(_T-_Dt)) ) { _t += _Dt; int substeps = _enrightTest2D.update(_t); COUT<<"substeps: "<<substeps<<ENDL; }
	glutSetWindow ( winID );
	glutPostRedisplay ();
}

void GLUTKeyboardDown( unsigned char key, int x, int y )
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
            _t += _Dt;
			_enrightTest2D.update(_t);
			break;
        }
		case 49: // 1
		{
			_isDisplayCircle = true;
			break;
		}
		case 50: // 2
		{
			_isDisplayCircle = false;
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
			_enrightTest2D.setAdvectionConditions( _cfl_number, _minSubSteps, _maxSubSteps, _Dt, _advScheme );
			COUT<<"cfl_number: "<<_cfl_number<<ENDL;
			break;
		}
		case GLUT_KEY_DOWN:
		{
			_cfl_number -= 0.1;
			_enrightTest2D.setAdvectionConditions( _cfl_number, _minSubSteps, _maxSubSteps, _Dt, _advScheme );
			COUT<<"cfl_number: "<<_cfl_number<<ENDL;
			break;
		}
    }
}

int main( int argc, char* argv[] )
{
	winX = winY;
	_enrightTest2D.initialize( _gridRes );
	_enrightTest2D.setAdvectionConditions( _cfl_number, _minSubSteps, _maxSubSteps, _Dt, _advScheme );

	// make circle to compare volume loss
	AABB2f aabb = AABB2f( Vec2f(0.f), Vec2f(1.f) );
	Grid2D grd = Grid2D( _gridRes, aabb );
	_circleLVS.initialize( grd, kUnified );
	const float Lx = aabb.width(0);
	const Vec2f minPt = aabb.minPoint();
	Vec2f center = Vec2f(0.5f*Lx, 0.75f*Lx) + minPt;
	float radius = 0.15*Lx;
	SetCircle( center, radius, _circleLVS );

    glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE );
	glutInitWindowSize( winX, winY );
    winID = glutCreateWindow( "EnrightTest2D" );
    glutDisplayFunc( Display );
    glutIdleFunc( Idle );
    glutKeyboardFunc( GLUTKeyboardDown );
	glutSpecialFunc( GLUTSpecialKeyboard );

    glutMainLoop();

    return 0;
}

