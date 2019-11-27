//----------------//
// ZalesakTest.cpp //
//-------------------------------------------------------//
// author: Julie Jang @ Dexter Studios                   //
// last update: 2018.04.10                               //
//-------------------------------------------------------//

#include <Bora.h>

using namespace Bora;

static float _cfl_number = 1.f;
static int _minSubSteps = 1;
static int _maxSubSteps = 2;
static AdvectionScheme _advScheme = kRK3;
static float _T = 6.28f, _Dt = _T/30.f;
static int winX = 0, winY = 800;
static int winID;
float 		   	 _t;
int				 _gridRes = 200;
ZalesakTest		 _zalesakTest;
bool			 _isDisplayZalesakSphere = false;
ScalarDenseField _zalesakSphereLVS; // import from cache file


void drawScalarField()
{
	glPointSize( (winX/_gridRes)+1.f );
	if( _isDisplayZalesakSphere )
	{
		int n = _gridRes;
		float minValue = _zalesakSphereLVS.minValue();
		AABB3f aabb = _zalesakSphereLVS.boundingBox();
		float L = aabb.width(0);
		Vec3f minPt = aabb.minPoint();
	
		glPointSize( 5 );
		glBegin( GL_POINTS );
		for( int k=0; k<n; k++ )
		for( int j=0; j<n; j++ )
		for( int i=0; i<n; i++ )
		{
			float value = _zalesakSphereLVS( i, j, k );
			if( value<EPSILON )
			{
				float normalizedValue = value/minValue;
				glColor( normalizedValue, 0.f, 0.f );
				Vec3f p = _zalesakSphereLVS.worldPoint( Vec3f(i, j, k) ) + Vec3f(L/(float)n/2.f);
				glVertex( p.x, p.y, p.z );
			}
		}
		glEnd();
	}
	else { _zalesakTest.drawScalarField(); }
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
	glMatrixMode ( GL_MODELVIEW );
	glLoadIdentity ();
	gluLookAt ( 0.0, 0.0, 50.f, 0.f, 0.f, 0.f, 0.0, 1.f, 0.0 );
	glClearColor ( 0.4f, 0.4f, 0.4f, 1.0f );
	glClear ( GL_COLOR_BUFFER_BIT );

    drawScalarField();

    glutSwapBuffers();
}

static void Idle ( void )
{
	if( (_t>=_Dt) && (_t<_T) ) { _t +=_Dt; _zalesakTest.update(_t); }
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
            _t+=_Dt;
			_zalesakTest.update(_t);
			break;
        }
		case 49: // 1
		{
			_isDisplayZalesakSphere = true;
			break;
		}
		case 50: // 2
		{
			_isDisplayZalesakSphere = false;
			break;
		}
    }
}

int main( int argc, char** argv )
{
	winX = winY;
	_zalesakTest.initialize( _gridRes );
	_zalesakTest.setAdvectionConditions( _cfl_number, _minSubSteps, _maxSubSteps, _Dt, _advScheme );

	// fill _zalesakSphereLVS
	AABB3f aabb = AABB3f( Vec3f(-50.f), Vec3f(50.f) );
	Grid grd = Grid( _gridRes, aabb );
	TriangleMesh triMesh;
	Vec3fArray& pos = triMesh.pos;
	IndexArray& ind = triMesh.indices;
	for( int i=0; i<369; ++i )
	{
		pos.append( ZalesakSphereVertices[i] );
	}
	for( int i=0; i<2202; ++i )
	{
		ind.append( ZalesakSphereIndices[i] );
	}
	MarkerDenseField stt;
	Voxelizer_FMM lvsSolver;
	_zalesakSphereLVS.initialize( grd, kUnified );
	stt.setGrid( grd );
	lvsSolver.set( grd );
	lvsSolver.addMesh( _zalesakSphereLVS, stt, triMesh, INFINITE, INFINITE );
	lvsSolver.finalize();

	glutInit( &argc, argv );
	glutInitDisplayMode ( GLUT_RGBA | GLUT_DOUBLE );
	glutInitWindowSize ( winX, winY );
	winID = glutCreateWindow ( "Zalesak Test" );
    glutDisplayFunc( Display );
    glutIdleFunc( Idle );
    glutKeyboardFunc( GLUTKeyboardDown );

	glutMainLoop();

	return 0;
}

