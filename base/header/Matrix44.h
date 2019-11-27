//------------//
// Matrix44.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2018.04.26                               //
//-------------------------------------------------------//

#ifndef _BoraMatrix44_h_
#define _BoraMatrix44_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

template <typename T>
class Matrix44
{
    public:

        union
        {
            T v[16];              // sixteen values
            Vector4<T> column[4]; // four column vectors
            struct                // _ij
            {
                T _00, _10, _20, _30; // row[0]
                T _01, _11, _21, _31; // row[1]
                T _02, _12, _22, _32; // row[2]
                T _03, _13, _23, _33; // row[3]
            };
        };

    public:

        BORA_FUNC_QUAL
        Matrix44()
        : v{ 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 }
        {}

        BORA_FUNC_QUAL
        Matrix44( const Matrix44& m )
        : v{ m._00,m._10,m._20,m._30, m._01,m._11,m._21,m._31, m._02,m._12,m._22,m._32, m._03,m._13,m._23,m._33 }
        {}

        BORA_FUNC_QUAL
        Matrix44
        (
            const T m00, const T m01, const T m02, const T m03,
            const T m10, const T m11, const T m12, const T m13,
            const T m20, const T m21, const T m22, const T m23,
            const T m30, const T m31, const T m32, const T m33
        )
        : v{ m00,m10,m20,m30, m01,m11,m21,m31, m02,m12,m22,m32, m03,m13,m23,m33 }
        {}

        BORA_FUNC_QUAL
        Matrix44( const T& s )
        : v{ s,s,s,s, s,s,s,s, s,s,s,s, s,s,s,s }
        {}

        BORA_FUNC_QUAL
        Matrix44( const T a[16] )
        : v{ a[0],a[1],a[2],a[3], a[4],a[5],a[6],a[7], a[8],a[9],a[10],a[11], a[12],a[13],a[14],a[15] }
        {}

        BORA_FUNC_QUAL
        Matrix44( const T x, const T y, const T z, const T w )
        : v{ x,0,0,0, 0,y,0,0, 0,0,z,0, 0,0,0,w }
        {}

        BORA_FUNC_QUAL
        Matrix44( const Vector3<T>& c0, const Vector3<T>& c1, const Vector3<T>& c2, const Vector4<T>& c3 )
        {
            column[0] = c0;
            column[1] = c1;
            column[2] = c2;
            column[3] = c3;
        }        

        BORA_FUNC_QUAL
        Matrix44( const Matrix33<T>& m )
        : v{ m._00, m._01, m._02, 0,
             m._10, m._11, m._12, 0,
             m._20, m._21, m._22, 0,
                 0,     0,     0, 1 }
        {}

        // OpenGL: column-major order
        BORA_FUNC_QUAL
        Matrix44( const glm::tmat4x4<T>& m )
        : column{ m[0], m[1], m[2], m[3] }
        {}

        BORA_FUNC_QUAL
        Matrix44& operator=( const Matrix44& m )
        {
            for( int i=0; i<16; ++i ) { v[i] = m.v[i]; }
            return (*this);
        }

        BORA_FUNC_QUAL
        Matrix44& set
        (
            const T m00, const T m01, const T m02, const T m03,
            const T m10, const T m11, const T m12, const T m13,
            const T m20, const T m21, const T m22, const T m23,
            const T m30, const T m31, const T m32, const T m33
        )
        {
            _00=m00; _01=m01; _02=m02; _03=m03;
            _10=m10; _11=m11; _12=m12; _13=m13;
            _20=m20; _21=m21; _22=m22; _23=m23;
            _30=m30; _31=m31; _32=m32; _33=m33;

            return (*this);
        }

        // the j-th column vector
        BORA_FUNC_QUAL
        Vector4<T>& operator[]( const int j )
        {
            return column[j];
        }

        // the j-th column vector
        BORA_FUNC_QUAL
        const Vector4<T>& operator[]( const int j ) const
        {
            return column[j];
        }

        // the (i,j)-element
        BORA_FUNC_QUAL
        T& operator()( const int i, const int j )
        {
            return column[j][i];
        }

        // the (i,j)-element
        BORA_FUNC_QUAL
        const T& operator()( const int i, const int j ) const
        {
            return column[j][i];
        }

        // the i-th row vector
        BORA_FUNC_QUAL
        Vector4<T> row( const int i ) const
        {
            return Vector4<T>( column[0][i], column[1][i], column[2][i], column[3][i] );
        }

        BORA_FUNC_QUAL
        bool operator==( const Matrix44& m )
        {
            for( int i=0; i<16; ++i )
            {
                if( v[i] != m.v[i] )
                {
                    return false;
                }
            }
            return true;
        }

        BORA_FUNC_QUAL
        bool operator!=( const Matrix44& m )
        {
            for( int i=0; i<16; ++i )
            {
                if( v[i] != m.v[i] )
                {
                    return true;
                }
            }
            return false;
        }

        BORA_FUNC_QUAL
        bool isSame( const Matrix44& m, const T tolerance=EPSILON ) const
        {
            for( int i=0; i<16; ++i )
            {
                if( !AlmostSame( v[i], m.v[i], tolerance ) )
                {
                    return false;
                }
            }

            return true;
        }

        BORA_FUNC_QUAL
        bool isSymmetric( const T tolerance=EPSILON ) const
        {
            if( !AlmostSame( _01, _10, tolerance ) ) { return false; }
            if( !AlmostSame( _02, _20, tolerance ) ) { return false; }
            if( !AlmostSame( _03, _30, tolerance ) ) { return false; }
            if( !AlmostSame( _12, _21, tolerance ) ) { return false; }
            if( !AlmostSame( _13, _31, tolerance ) ) { return false; }
            if( !AlmostSame( _23, _32, tolerance ) ) { return false; }

            return true;
        }

        BORA_FUNC_QUAL
        bool isIdentity( const T tolerance=EPSILON ) const
        {
            for( int i=0; i<4; ++i )
            for( int j=0; j<4; ++j )
            {{
                if( i == j )
                {
                    if( !AlmostSame( column[i][j], (T)1, tolerance ) )
                    {
                        return false;
                    }
                }
                else
                {
                    if( !AlmostZero( column[i][j], tolerance ) )
                    {
                        return false;
                    }
                }
            }}

            return true;
        }

        BORA_FUNC_QUAL
        Matrix44& operator+=( const Matrix44& m )
        {
            for( int i=0; i<16; ++i ) { v[i] += m.v[i]; }
            return (*this);
        }

        BORA_FUNC_QUAL
        Matrix44& operator-=( const Matrix44& m )
        {
            for( int i=0; i<16; ++i ) { v[i] -= m.v[i]; }
            return (*this);
        }

        BORA_FUNC_QUAL
        Matrix44& operator*=( const T s )
        {
            for( int i=0; i<16; ++i ) { v[i] *= s; }
            return *this;
        }

        BORA_FUNC_QUAL
        Matrix44& operator*=( const Matrix44& m )
        {
            Matrix44 tmp( *this );

            for( int j=0; j<4; ++j )
            for( int i=0; i<4; ++i )
            {{
                T& d = column[j][i] = 0;

                for( int k=0; k<4; ++k )
                {
                    d += tmp.column[k][i] * m.column[j][k];
                }
            }}

            return (*this);
        }

        BORA_FUNC_QUAL
        Matrix44 operator*( const T s ) const
        {
            return ( Matrix44(*this) *= s );
        }

        BORA_FUNC_QUAL
        Matrix44 operator*( const Matrix44& m ) const
        {
            return ( Matrix44(*this) *= m );
        }

        BORA_FUNC_QUAL
        Vector4<T> operator*( const Vector4<T>& v ) const
        {
            Vector4<T> tmp;
            tmp.x = _00 * v.x + _01 * v.y + _02 * v.z + _03 * v.w;
            tmp.y = _10 * v.x + _11 * v.y + _12 * v.z + _13 * v.w;
            tmp.z = _20 * v.x + _21 * v.y + _22 * v.z + _23 * v.w;
            tmp.w = _30 * v.x + _31 * v.y + _32 * v.z + _33 * v.w;
            return tmp;
        }

        BORA_FUNC_QUAL
        Matrix44 operator+( const Matrix44& m ) const
        {
            Matrix44 tmp;
            for( int i=0; i<16; ++i ) { tmp.v[i] = v[i] + m.v[i]; }
            return tmp;
        }

        BORA_FUNC_QUAL
        Matrix44 operator-( const Matrix44& m ) const
        {
            Matrix44 tmp;
            for( int i=0; i<16; ++i ) { tmp.v[i] = v[i] - m.v[i]; }
            return tmp;
        }

        BORA_FUNC_QUAL
        operator glm::tmat4x4<T>() const
        {
            return glm::tmat4x4<T>( column[0], column[1], column[2], column[3] );
        }

        BORA_FUNC_QUAL
        void setUpper33( const Matrix33<T>& m )
        {
            _00=m._00; _01=m._01; _02=m._02;
            _10=m._10; _11=m._11; _12=m._12;
            _20=m._20; _21=m._21; _22=m._22;
        }

        BORA_FUNC_QUAL
        void getUpper33( const Matrix33<T>& m ) const
        {
            m._00=_00; m._01=_01; m._02=_02;
            m._10=_10; m._11=_11; m._12=_12;
            m._20=_20; m._21=_21; m._22=_22;
        }

        BORA_FUNC_QUAL
        Matrix44& transpose()
        {
            Swap(_01,_10); Swap(_02,_20); Swap(_03,_30);
            Swap(_12,_21); Swap(_13,_31);
            Swap(_23,_32);
            return (*this);
        }

        BORA_FUNC_QUAL
        Matrix44 transposed() const
        {
            Matrix44 tmp( *this );
            tmp.transpose();
            return tmp;
        }

        BORA_FUNC_QUAL
        T trace() const
        {
            return ( _00 + _11 + _22 + _33 );
        }

        BORA_FUNC_QUAL
        double determinant() const
        {
            return double
            (
                _03 * _12 * _21 * _30 - _02 * _13 * _21 * _30 -
                _03 * _11 * _22 * _30 + _01 * _13 * _22 * _30 +
                _02 * _11 * _23 * _30 - _01 * _12 * _23 * _30 -
                _03 * _12 * _20 * _31 + _02 * _13 * _20 * _31 +
                _03 * _10 * _22 * _31 - _00 * _13 * _22 * _31 -
                _02 * _10 * _23 * _31 + _00 * _12 * _23 * _31 +
                _03 * _11 * _20 * _32 - _01 * _13 * _20 * _32 -
                _03 * _10 * _21 * _32 + _00 * _13 * _21 * _32 +
                _01 * _10 * _23 * _32 - _00 * _11 * _23 * _32 -
                _02 * _11 * _20 * _33 + _01 * _12 * _20 * _33 +
                _02 * _10 * _21 * _33 - _00 * _12 * _21 * _33 -
                _01 * _10 * _22 * _33 + _00 * _11 * _22 * _33
            );
        }

        BORA_FUNC_QUAL
        Matrix44& inverse()
        {
            const double _det = 1.0 / ( Matrix44::determinant() + EPSILON );

            const double m00=_00, m01=_01, m02=_02, m03=_03;
            const double m10=_10, m11=_11, m12=_12, m13=_13;
            const double m20=_20, m21=_21, m22=_22, m23=_23;
            const double m30=_30, m31=_31, m32=_32, m33=_33;

            _00 = (T)(  ( m11*(m22*m33-m23*m32) - m12*(m21*m33-m23*m31) + m13*(m21*m32-m22*m31) ) * _det );
            _01 = (T)( -( m01*(m22*m33-m23*m32) - m02*(m21*m33-m23*m31) + m03*(m21*m32-m22*m31) ) * _det );
            _02 = (T)(  ( m01*(m12*m33-m13*m32) - m02*(m11*m33-m13*m31) + m03*(m11*m32-m12*m31) ) * _det );
            _03 = (T)( -( m01*(m12*m23-m13*m22) - m02*(m11*m23-m13*m21) + m03*(m11*m22-m12*m21) ) * _det );

            _10 = (T)( -( m10*(m22*m33-m23*m32) - m12*(m20*m33-m23*m30) + m13*(m20*m32-m22*m30) ) * _det );
            _11 = (T)(  ( m00*(m22*m33-m23*m32) - m02*(m20*m33-m23*m30) + m03*(m20*m32-m22*m30) ) * _det );
            _12 = (T)( -( m00*(m12*m33-m13*m32) - m02*(m10*m33-m13*m30) + m03*(m10*m32-m12*m30) ) * _det );
            _13 = (T)(  ( m00*(m12*m23-m13*m22) - m02*(m10*m23-m13*m20) + m03*(m10*m22-m12*m20) ) * _det );

            _20 = (T)(  ( m10*(m21*m33-m23*m31) - m11*(m20*m33-m23*m30) + m13*(m20*m31-m21*m30) ) * _det );
            _21 = (T)( -( m00*(m21*m33-m23*m31) - m01*(m20*m33-m23*m30) + m03*(m20*m31-m21*m30) ) * _det );
            _22 = (T)(  ( m00*(m11*m33-m13*m31) - m01*(m10*m33-m13*m30) + m03*(m10*m31-m11*m30) ) * _det );
            _23 = (T)( -( m00*(m11*m23-m13*m21) - m01*(m10*m23-m13*m20) + m03*(m10*m21-m11*m20) ) * _det );

            _30 = (T)( -( m10*(m21*m32-m22*m31) - m11*(m20*m32-m22*m30) + m12*(m20*m31-m21*m30) ) * _det );
            _31 = (T)(  ( m00*(m21*m32-m22*m31) - m01*(m20*m32-m22*m30) + m02*(m20*m31-m21*m30) ) * _det );
            _32 = (T)( -( m00*(m11*m32-m12*m31) - m01*(m10*m32-m12*m30) + m02*(m10*m31-m11*m30) ) * _det );
            _33 = (T)(  ( m00*(m11*m22-m12*m21) - m01*(m10*m22-m12*m20) + m02*(m10*m21-m11*m20) ) * _det );

            return (*this);
        }

        BORA_FUNC_QUAL
        Matrix44 inversed() const
        {
            return Matrix44(*this).inverse();
        }

        BORA_FUNC_QUAL
        void identity()
        {
            _00=1; _01=0; _02=0; _03=0;
            _10=0; _11=1; _12=0; _13=0;
            _20=0; _21=0; _22=1; _23=0;
            _30=0; _31=0; _32=1; _33=0;
        }

        BORA_FUNC_QUAL
        Vector3<T> transform( const Vector3<T>& v, bool asVector ) const
        {
            const float& x = v.x;
            const float& y = v.y;
            const float& z = v.z;

            Vector3<T> tmp( _00*x+_01*y+_02*z, _10*x+_11*y+_12*z, _20*x+_21*y+_22*z );
            if( asVector ) { return tmp; } // no consideration for translation

            tmp.x += _03;
            tmp.y += _13;
            tmp.z += _23;

            return tmp;
        }

        BORA_FUNC_QUAL
        Vector3<T> transform( const Vector3<T>& v, const Vector3<T>& pivot, bool asVector ) const
        {
            Vector3<T> tmp( v.x-pivot.x, v.y-pivot.y, v.z-pivot.z );
            Matrix44::transform( tmp, asVector );
            tmp += pivot;
            return tmp;
        }

        BORA_FUNC_QUAL
        void setTranslation( const T tx, const T ty, const T tz )
        {
            _03 = tx;
            _13 = ty;
            _23 = tz;
        }

        BORA_FUNC_QUAL
        void setTranslation( const Vector3<T>& t )
        {
            Matrix44::setTranslation( t.x, t.y, t.z );
        }

        BORA_FUNC_QUAL
        void getTranslation( T& tx, T& ty, T& tz )
        {
            tx = _03;
            ty = _13;
            tz = _23;
        }

        BORA_FUNC_QUAL
        void getTranslation( Vector3<T>& t )
        {
            Matrix44::getTranslation( t.x, t.y, t.z );
        }

        void setRotationAboutX( const T radians )
        {
            const T s = Sin( radians );
            const T c = Cos( radians );

            _00=1; _01=0; _02=0;
            _10=0; _11=c; _12=-s;
            _20=0; _21=s; _22=c;
        }

        void setRotationAboutY( const T radians )
        {
            const T s = Sin( radians );
            const T c = Cos( radians );

            _00=c;  _01=0; _02=s;
            _10=0;  _11=1; _12=0;
            _20=-s; _21=0; _22=c;
        }

        void setRotationAboutZ( const T radians )
        {
            const T s = Sin( radians );
            const T c = Cos( radians );

            _00=c; _01=-s; _02=0;
            _10=s; _11=c;  _12=0;
            _20=0; _21=0;  _22=1;
        }

        // in radians
        // default order: X>Y>Z order: q = (Rz*Ry*Rx) * p
        void setRotation( const T rx, const T ry, const T rz, const RotationOrder order=kZYX )
        {
            const T sx=Sin(rx), cx=Cos(rx);
            const T sy=Sin(ry), cy=Cos(ry);
            const T sz=Sin(rz), cz=Cos(rz);

            if( order == RotationOrder::kXYZ )
            {
                _00=cy*cz;          _01=-cy*sz;         _02=sy;
                _10=cx*sz+sx*sy*cz; _11=cx*cz-sx*sy*sz; _12=-sx*cy;
                _20=sx*sz-cx*sy*cz; _21=sx*cz+cx*sy*sz; _22=cx*cy;

                return;
            }

            if( order == RotationOrder::kYZX )
            {
                _00=cy*cz;          _01=sx*sy-cx*cy*sz; _02=sx*cy*sz+cx*sy;
                _10=sz;             _11=cx*cz;          _12=-sx*cz;
                _20=-sy*cz;         _21=cx*sy*sz+sx*cy; _22=cx*cy-sx*sy*sz;

                return;
            }

            if( order == RotationOrder::kZXY )
            {
                _00=cy*cz-sx*sy*sz; _01=-cx*sz;         _02=sy*cz+sx*cy*sz;
                _10=cy*sz+sx*sy*cz; _11=cx*cz;          _12=sy*sz-sx*cy*cz;
                _20=-cx*sy;         _21=sx;             _22=cx*cy;

                return;
            }

            if( order == RotationOrder::kXZY )
            {
                _00=cy*cz;          _01=-sz;            _02=sy*cz;
                _10=cx*cy*sz+sx*sy; _11=cx*cz;          _12=cx*sy*sz-sx*cy;
                _20=sx*cy*sz-cx*sy; _21=sx*cz;          _22=sx*sy*sz+cx*cy;

                return;
            }

            if( order == RotationOrder::kYXZ )
            {
                _00=cy*cz+sx*sy*sz; _01=sx*sy*cz-cy*sz; _02=cx*sy;
                _10=cx*sz;          _11=cx*cz;          _12=-sx;
                _20=sx*cy*sz-sy*cz; _21=sy*sz+sx*cy*cz; _22=cx*cy;

                return;
            }

            if( order == RotationOrder::kZYX )
            {
                _00=cy*cz;          _01=sx*sy*cz-cx*sz; _02=cx*sy*cz+sx*sz;
                _10=cy*sz;          _11=sx*sy*sz+cx*cz; _12=cx*sy*sz-sx*cz;
                _20=-sy;            _21=sx*cy;          _22=cx*cy;

                return;
            }
        }

        // in radians
        void setRotation( const Vector3<T>& angles, const RotationOrder order=kZYX )
        {
            Matrix44::setRotation( angles.x, angles.y, angles.z, order );
        }

        // in radians
        // X>Y>Z order: q = (Rz*Ry*Rx) * p
        // The matrix must be orthonormal before executing this function.
        // reference: "Extracting Euler Angles from a Rotation Matrix"
        void getRotation( T& rx, T& ry, T& rz, const RotationOrder order=kZYX ) const
        {
            if( order == RotationOrder::kXYZ )
            {
                COUT << "Error@Matrix44::getRotation(): Not yet implemented." << ENDL;
                return;
            }

            if( order == RotationOrder::kYZX )
            {
                COUT << "Error@Matrix44::getRotation(): Not yet implemented." << ENDL;
                return;
            }

            if( order == RotationOrder::kZXY )
            {
                COUT << "Error@Matrix44::getRotation(): Not yet implemented." << ENDL;
                return;
            }

            if( order == RotationOrder::kXZY )
            {
                COUT << "Error@Matrix44::getRotation(): Not yet implemented." << ENDL;
                return;
            }

            if( order == RotationOrder::kYXZ )
            {
                COUT << "Error@Matrix44::getRotation(): Not yet implemented." << ENDL;
                return;
            }

            if( order == RotationOrder::kZYX )
            {
                const T c2 = Sqrt( _00*_00 + _10*_10 );

                rx = ATan2( _21, _22 );
                ry = ATan2( -_20, c2 );

                const T s = Sin( rx );
                const T c = Cos( rx );

                rz = ATan2( s*_02-c*_01, c*_11-s*_12 );

                return;
            }
        }

        // in radians
        void getRotation( Vector3<T>& r ) const
        {
            Matrix44::getRotation( r.x, r.y, r.z );
        }

        void eliminateScale()
        {
            column[0].normalize();
            column[1].normalize();
            column[2].normalize();
        }

        void addScale( const T sx, const T sy, const T sz )
        {
            _00*=sx;  _01*=sy;  _02*=sz;
            _10*=sx;  _11*=sy;  _12*=sz;
            _20*=sx;  _21*=sy;  _22*=sz;
        }

        void setScale( const T sx, const T sy, const T sz )
        {
            Matrix44::eliminateScale();
            Matrix44::addScale( sx, sy, sz );
        }

        void setScale( const Vector3<T>& s )
        {
            Matrix44::setScale( s.x, s.y, s.z );
        }

        void getScale( T& sx, T& sy, T& sz ) const
        {
            sx = column[0].length();
            sy = column[1].length();
            sz = column[2].length();
        }

        void getScale( Vector3<T>& s ) const
        {
            Matrix44::getScale( s.x, s.y, s.z );
        }

        void decompose( Vector3<T>& t, Vector3<T>& r, Vector3<T>& s ) const
        {
            Matrix44 tmp( *this );

            tmp.getTranslation( t );
            tmp.getScale( s );
            tmp.eliminateScale();
            tmp.getRotation( r );
        }

        void write( std::ofstream& fout ) const
        {
            fout.write( (char*)v, 16*sizeof(T) );
        }

        void read( std::ifstream& fin )
        {
            fin.read( (char*)v, 16*sizeof(T) );
        }

        bool save( const char* filePathName ) const
        {
            std::ofstream fout( filePathName, std::ios::out|std::ios::binary );

            if( fout.fail() )
            {
                COUT << "Error@Matrix44::save(): Failed to open file " << filePathName << ENDL;
                return false;
            }

            Matrix44::write( fout );

            return true;
        }

        bool load( const char* filePathName )
        {
            std::ifstream fin( filePathName, std::ios::out|std::ios::binary );

            if( fin.fail() )
            {
                COUT << "Error@Matrix44::save(): Failed to open file " << filePathName << ENDL;
                return false;
            }

            Matrix44::write( fin );

            return true;
        }
};

template <typename T>
BORA_FUNC_QUAL
inline Matrix44<T> operator*( const T s, const Matrix44<T>& m )
{
    Matrix44<T> tmp( m );
    for( int i=0; i<16; ++i ) { tmp.v[i] *= s; }
    return tmp;
}

template <typename T>
inline std::ostream& operator<<( std::ostream& os, const Matrix44<T>& m )
{
    std::string ret;
    std::string indent;

    const int indentation = 0;
    indent.append( indentation+1, ' ' );

    ret.append( "[" );

    for( int i=0; i<4; ++i )
    {
        ret.append( "[" );

        for( int j=0; j<4; ++j )
        {
            if( j ) { ret.append(", "); }
            ret.append( std::to_string( m(i,j) ) );
        }

        ret.append("]");

        if( i< 4-1 )
        {
            ret.append( ",\n" );
            ret.append( indent );
        }
    }

    ret.append( "]" );

    os << ret;

    return os;
}

typedef Matrix44<float>  Mat44f;
typedef Matrix44<double> Mat44d;

BORA_NAMESPACE_END

#endif

