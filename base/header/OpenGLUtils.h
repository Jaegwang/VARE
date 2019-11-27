//---------------//
// OpenGLUtils.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.04.25                               //
//-------------------------------------------------------//

#ifndef _BoraOpenGLUtils_h_
#define _BoraOpenGLUtils_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

inline void glVertex( const float x, const float y ) { glVertex2f( x, y ); }
inline void glVertex( const double x, const double y ) { glVertex2d( x, y ); }

inline void glVertex( const float x, const float y, const float z ) { glVertex3f( x, y, z ); }
inline void glVertex( const double x, const double y, const double z ) { glVertex3d( x, y, z ); }

inline void glVertex( const Vec2f& p ) { glVertex2fv( &p.x ); }
inline void glVertex( const Vec2d& p ) { glVertex2dv( &p.x ); }
inline void glVertex( const Vec3f& p ) { glVertex3fv( &p.x ); }
inline void glVertex( const Vec3d& p ) { glVertex3dv( &p.x ); }
inline void glVertex( const Vec4f& p ) { glVertex4fv( &p.x ); }
inline void glVertex( const Vec4d& p ) { glVertex4dv( &p.x ); }

inline void glColor( const float c ) { glColor3f( c, c, c ); }
inline void glColor( const float r, const float g, const float b ) { glColor3f( r, g, b ); }
inline void glColor( const double r, const double g, const double b ) { glColor3d( r, g, b ); }
inline void glColor( const Vec3f& p ) { glColor3fv( &p.r ); }
inline void glColor( const Vec3d& p ) { glColor3dv( &p.r ); }
inline void glColor( const Vec4f& p ) { glColor4fv( &p.r ); }
inline void glColor( const Vec4d& p ) { glColor4dv( &p.r ); }

inline void glNormal( const float x, const float y, const float z ) { glNormal3f( x, y, z ); }
inline void glNormal( const double x, const double y, const double z ) { glNormal3d( x, y, z ); }
inline void glNormal( const Vec3f& p ) { glNormal3fv( &p.x ); }
inline void glNormal( const Vec3d& p ) { glNormal3dv( &p.x ); }

inline void glSegment( const Vec3f& p, const Vec3f& q ) { glVertex(p); glVertex(q); }
inline void glRay( const Vec3f& p, const Vec3f& v ) { glVertex(p); glVertex(p+v); }

inline void glTranslate( const float tx, const float ty, const float tz ) { glTranslatef( tx, ty, tz ); }
inline void glTranslate( const double tx, const double ty, const double tz ) { glTranslated( tx, ty, tz ); }
inline void glTranslate( const Vec3f& t ) { glTranslatef( t.x, t.y, t.z ); }
inline void glTranslate( const Vec3d& t ) { glTranslated( t.x, t.y, t.z ); }

inline void glScale( const float s ) { glScalef( s, s, s ); }
inline void glScale( const double s ) { glScaled( s, s, s ); }
inline void glScale( const float sx, const float sy, const float sz ) { glScalef( sx, sy, sz ); }
inline void glScale( const double sx, const double sy, const double sz ) { glScaled( sx, sy, sz ); }
inline void glScale( const Vec3f& s ) { glScalef( s.x, s.y, s.z ); }
inline void glScale( const Vec3d& s ) { glScaled( s.x, s.y, s.z ); }

inline void DrawCube( const Vec3f& minPt, const Vec3f& maxPt )
{
    glBegin( GL_LINES );
    {
        // bottom
        glVertex(minPt.x,minPt.y,minPt.z); glVertex(maxPt.x,minPt.y,minPt.z);
        glVertex(maxPt.x,minPt.y,minPt.z); glVertex(maxPt.x,minPt.y,maxPt.z);
        glVertex(maxPt.x,minPt.y,maxPt.z); glVertex(minPt.x,minPt.y,maxPt.z);
        glVertex(minPt.x,minPt.y,maxPt.z); glVertex(minPt.x,minPt.y,minPt.z);

        // top
        glVertex(minPt.x,maxPt.y,minPt.z); glVertex(maxPt.x,maxPt.y,minPt.z);
        glVertex(maxPt.x,maxPt.y,minPt.z); glVertex(maxPt.x,maxPt.y,maxPt.z);
        glVertex(maxPt.x,maxPt.y,maxPt.z); glVertex(minPt.x,maxPt.y,maxPt.z);
        glVertex(minPt.x,maxPt.y,maxPt.z); glVertex(minPt.x,maxPt.y,minPt.z);

        // four vertical lines
        glVertex(minPt.x,minPt.y,minPt.z); glVertex(minPt.x,maxPt.y,minPt.z);
        glVertex(maxPt.x,minPt.y,minPt.z); glVertex(maxPt.x,maxPt.y,minPt.z);
        glVertex(maxPt.x,minPt.y,maxPt.z); glVertex(maxPt.x,maxPt.y,maxPt.z);
        glVertex(minPt.x,minPt.y,maxPt.z); glVertex(minPt.x,maxPt.y,maxPt.z);
    }
    glEnd();
}

BORA_NAMESPACE_END

#endif

