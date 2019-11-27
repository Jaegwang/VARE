//------------//
// Matrix33.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
//         Wanho Choi @ Dexter Studios                   //
//         Julie Jang @ Dexter Studios                   //
// last update: 2018.04.26                               //
//-------------------------------------------------------//

#ifndef _BoraMatrix33_h_
#define _BoraMatrix33_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

template <typename T>
class Matrix33
{
    public:

        union
        {
            T v[9];               // nine values
            Vector3<T> column[3]; // three column vectors
            struct                // _ij
            {
                T _00, _10, _20; // row[0]
                T _01, _11, _21; // row[1]
                T _02, _12, _22; // row[2]
            };
        };

    public:

        BORA_FUNC_QUAL
        Matrix33()
        : v{ 1,0,0, 0,1,0, 0,0,1 }
        {}

        BORA_FUNC_QUAL
        Matrix33( const Matrix33& m )
        : v{ m._00,m._10,m._20, m._01,m._11,m._21, m._02,m._12,m._22 }
        {}

        BORA_FUNC_QUAL
        Matrix33
        (
            const T m00, const T m01, const T m02,
            const T m10, const T m11, const T m12,
            const T m20, const T m21, const T m22
        )
        : v{ m00,m10,m20, m01,m11,m21, m02,m12,m22 }
        {}

        BORA_FUNC_QUAL
        Matrix33( const T& s )
        : v{ s,s,s, s,s,s, s,s,s }
        {}

        BORA_FUNC_QUAL
        Matrix33( const T a[9] )
        : v{ a[0],a[1],a[2], a[3],a[4],a[5], a[6],a[7],a[8] }
        {}

        BORA_FUNC_QUAL
        Matrix33( const T x, const T y, const T z )
        : v{ x,0,0, 0,y,0, 0,0,z }
        {}

        BORA_FUNC_QUAL
        Matrix33( const Vector3<T>& c0, const Vector3<T>& c1, const Vector3<T>& c2 )
        {
            column[0] = c0;
            column[1] = c1;
            column[2] = c2;
        }

        BORA_FUNC_QUAL
        Matrix33( const Matrix22<T>& m )
        : v{ m._00, m._01, 0,
             m._10, m._11, 0,
                 0,     0, 1 }
        {}

        // OpenGL: column-major order
        BORA_FUNC_QUAL
        Matrix33( const glm::tmat3x3<T>& m )
        : column{ m[0], m[1], m[2] }
        {}

        BORA_FUNC_QUAL
        Matrix33& operator=( const Matrix33& m )
        {
            for( int i=0; i<9; ++i ) { v[i] = m.v[i]; }
            return (*this);
        }

        BORA_FUNC_QUAL
        Matrix33& set
        (
            const T m00, const T m01, const T m02,
            const T m10, const T m11, const T m12,
            const T m20, const T m21, const T m22
        )
        {
            _00=m00; _01=m01; _02=m02;
            _10=m10; _11=m11; _12=m12;
            _20=m20; _21=m21; _22=m22;

            return (*this);
        }

        // the j-th column vector
        BORA_FUNC_QUAL
        Vector3<T>& operator[]( const int j )
        {
            return column[j];
        }

        // the j-th column vector
        BORA_FUNC_QUAL
        const Vector3<T>& operator[]( const int j ) const
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
        Vector3<T> row( const int i ) const
        {
            return Vector3<T>( column[0][i], column[1][i], column[2][i] );
        }

        BORA_FUNC_QUAL
        bool operator==( const Matrix33& m )
        {
            for( int i=0; i<9; ++i )
            {
                if( v[i] != m.v[i] )
                {
                    return false;
                }
            }
            return true;
        }

        BORA_FUNC_QUAL
        bool operator!=( const Matrix33& m )
        {
            for( int i=0; i<9; ++i )
            {
                if( v[i] != m.v[i] )
                {
                    return true;
                }
            }
            return false;
        }

        BORA_FUNC_QUAL
        bool isSame( const Matrix33& m, const T tolerance=EPSILON ) const
        {
            for( int i=0; i<9; ++i )
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
            if( !AlmostSame( _12, _21, tolerance ) ) { return false; }

            return true;
        }

        BORA_FUNC_QUAL
        bool isIdentity( const T tolerance=EPSILON ) const
        {
            for( int i=0; i<3; ++i )
            for( int j=0; j<3; ++j )
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
        Matrix33& operator+=( const Matrix33& m )
        {
            for( int i=0; i<9; ++i ) { v[i] += m.v[i]; }
            return (*this);
        }

        BORA_FUNC_QUAL
        Matrix33& operator-=( const Matrix33& m )
        {
            for( int i=0; i<9; ++i ) { v[i] -= m.v[i]; }
            return (*this);
        }

        BORA_FUNC_QUAL
        Matrix33& operator*=( const T s )
        {
            for( int i=0; i<9; ++i ) { v[i] *= s; }
            return *this;
        }

        BORA_FUNC_QUAL
        Matrix33& operator*=( const Matrix33& m )
        {
            Matrix33 tmp( *this );

            for( int j=0; j<3; ++j )
            for( int i=0; i<3; ++i )
            {{
                T& d = column[j][i] = 0;

                for( int k=0; k<3; ++k )
                {
                    d += tmp.column[k][i] * m.column[j][k];
                }
            }}

            return (*this);
        }

        BORA_FUNC_QUAL
        Matrix33 operator*( const T s ) const
        {
            return ( Matrix33(*this) *= s );
        }

        BORA_FUNC_QUAL
        Matrix33 operator*( const Matrix33& m ) const
        {
            return ( Matrix33(*this) *= m );
        }

        BORA_FUNC_QUAL
        Vector3<T> operator*( const Vector3<T>& v ) const
        {
            Vector3<T> tmp;
            tmp.x = _00 * v.x + _01 * v.y + _02 * v.z;
            tmp.y = _10 * v.x + _11 * v.y + _12 * v.z;
            tmp.z = _20 * v.x + _21 * v.y + _22 * v.z;
            return tmp;
        }

        BORA_FUNC_QUAL
        Matrix33 operator+( const Matrix33& m ) const
        {
            Matrix33 tmp;
            for( int i=0; i<9; ++i ) { tmp.v[i] = v[i] + m.v[i]; }
            return tmp;
        }

        BORA_FUNC_QUAL
        Matrix33 operator-( const Matrix33& m ) const
        {
            Matrix33 tmp;
            for( int i=0; i<9; ++i ) { tmp.v[i] = v[i] - m.v[i]; }
            return tmp;
        }

        BORA_FUNC_QUAL
        operator glm::tmat3x3<T>() const
        {
            return glm::tmat3x3<T>( column[0], column[1], column[2] );
        }        

        BORA_FUNC_QUAL
        void setUpper22( const Matrix22<T>& m )
        {
            _00=m._00; _01=m._01;
            _10=m._10; _11=m._11;
        }

        BORA_FUNC_QUAL
        void getUpper22( const Matrix22<T>& m ) const
        {
            m._00=_00; m._01=_01;
            m._10=_10; m._11=_11;
        }

        BORA_FUNC_QUAL
        Matrix33& transpose()
        {
            Swap(_01,_10); Swap(_02,_20);
            Swap(_12,_21);
            return (*this);
        }

        BORA_FUNC_QUAL
        Matrix33 transposed() const
        {
            Matrix33 tmp( *this );
            tmp.transpose();
            return tmp;
        }

        BORA_FUNC_QUAL
        T trace() const
        {
            return ( _00 + _11 + _22 );
        }

        BORA_FUNC_QUAL
        T cofactor( const int i, const int j ) const
        {
            const Matrix33& A = *this;

            const T a = A( (i+1)%3, (j+1)%3 );
            const T b = A( (i+1)%3, (j+2)%3 );
            const T c = A( (i+2)%3, (j+1)%3 );
            const T d = A( (i+2)%3, (j+2)%3 );

            return ( a*d - b*c );
        }

        BORA_FUNC_QUAL
        double determinant() const
        {
            return double
            (
                _00 * ( _11 * _22 - _21 * _12 ) - 
                _01 * ( _10 * _22 - _12 * _20 ) + 
                _02 * ( _10 * _21 - _11 * _20 )
            );
        }

        BORA_FUNC_QUAL
        Matrix33& inverse()
        {
            const double _det = 1.0 / ( Matrix33::determinant() + EPSILON );

            const double m00=_00, m01=_01, m02=_02;
            const double m10=_10, m11=_11, m12=_12;
            const double m20=_20, m21=_21, m22=_22;

            _00 = (T)( ( m11*m22 - m12*m21 ) * _det );
            _01 = (T)( ( m02*m21 - m01*m22 ) * _det );
            _02 = (T)( ( m01*m12 - m02*m11 ) * _det );

            _10 = (T)( ( m12*m20 - m10*m22 ) * _det );
            _11 = (T)( ( m00*m22 - m02*m20 ) * _det );
            _12 = (T)( ( m02*m10 - m00*m12 ) * _det );

            _20 = (T)( ( m10*m21 - m11*m20 ) * _det );
            _21 = (T)( ( m01*m20 - m00*m21 ) * _det );
            _22 = (T)( ( m00*m11 - m01*m10 ) * _det );

            return (*this);
        }

        BORA_FUNC_QUAL
        Matrix33 inversed() const
        {
            return Matrix33(*this).inverse();
        }

        BORA_FUNC_QUAL
        void identity()
        {
            _00=1; _01=0; _02=0;
            _10=0; _11=1; _12=0;
            _20=0; _21=0; _22=1;
        }

        BORA_FUNC_QUAL
        Vector2<T> transform( const Vector2<T>& v, bool asVector ) const
        {
            const float& x = v.x;
            const float& y = v.y;

            Vector2<T> tmp( _00*x+_01*y, _10*x+_11*y );
            if( asVector ) { return tmp; } // no consideration for translation

            tmp.x += _02;
            tmp.y += _12;

            return tmp;
        }

        BORA_FUNC_QUAL
        Vector2<T> transform( const Vector2<T>& v, const Vector2<T>& pivot, bool asVector ) const
        {
            Vector2<T> tmp( v.x-pivot.x, v.y-pivot.y );
            Matrix33::transform( tmp, asVector );
            tmp += pivot;
            return tmp;
        }

        BORA_FUNC_QUAL
        void setTranslation( const T tx, const T ty )
        {
            _02 = tx;
            _12 = ty;
        }

        BORA_FUNC_QUAL
        void setTranslation( const Vector2<T>& t )
        {
            Matrix33::setTranslation( t.x, t.y );
        }

        BORA_FUNC_QUAL
        void getTranslation( T& tx, T& ty )
        {
            tx = _02;
            ty = _12;
        }

        BORA_FUNC_QUAL
        void getTranslation( Vector2<T>& t )
        {
            Matrix33::getTranslation( t.x, t.y );
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
            Matrix33::setRotation( angles.x, angles.y, angles.z, order );
        }

        // in radians
        // X>Y>Z order: q = (Rz*Ry*Rx) * p
        // The matrix must be orthonormal before executing this function.
        // reference: "Extracting Euler Angles from a Rotation Matrix"
        void getRotation( T& rx, T& ry, T& rz, const RotationOrder order=kZYX ) const
        {
            if( order == RotationOrder::kXYZ )
            {
                COUT << "Error@Matrix33::getRotation(): Not yet implemented." << ENDL;
                return;
            }

            if( order == RotationOrder::kYZX )
            {
                COUT << "Error@Matrix33::getRotation(): Not yet implemented." << ENDL;
                return;
            }

            if( order == RotationOrder::kZXY )
            {
                COUT << "Error@Matrix33::getRotation(): Not yet implemented." << ENDL;
                return;
            }

            if( order == RotationOrder::kXZY )
            {
                COUT << "Error@Matrix33::getRotation(): Not yet implemented." << ENDL;
                return;
            }

            if( order == RotationOrder::kYXZ )
            {
                COUT << "Error@Matrix33::getRotation(): Not yet implemented." << ENDL;
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
            Matrix33::getRotation( r.x, r.y, r.z );
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
            Matrix33::eliminateScale();
            Matrix33::addScale( sx, sy, sz );
        }

        void setScale( const Vector3<T>& s )
        {
            Matrix33::setScale( s.x, s.y, s.z );
        }

        void getScale( T& sx, T& sy, T& sz ) const
        {
            sx = column[0].length();
            sy = column[1].length();
            sz = column[2].length();
        }

        void getScale( Vector3<T>& s ) const
        {
            Matrix33::getScale( s.x, s.y, s.z );
        }

        void decompose( Vector3<T>& r, Vector3<T>& s ) const
        {
            Matrix33 tmp( *this );

            tmp.getScale( s );
            tmp.eliminateScale();
            tmp.getRotation( r );
        }

        // Two eigenvectors are orthonormal each other only when the matrix is symmetric.
        bool eigen( T eigenvalues[3], Vector3<T> eigenvectors[3] ) const
        {
            //eigen( eigenvalues[0], eigenvalues[1], eigenvalues[2], eigenvectors[0], eigenvectors[1], eigenvectors[2] );

            const Matrix33& A = *this;

            // eigenvalues
            {
                const T a = 1;
                const T b = -A.trace();
                const T c = A.cofactor(0,0) + A.cofactor(1,1) + A.cofactor(2,2);
                const T d = -A.determinant();

                if( SolveCubicEqn( a,b,c,d, eigenvalues ) != 3 )
                {
                    COUT << "Error@Matrix33::eigen(): Invalid case." << ENDL;
                    return false;
                }

                Sort( eigenvalues[0], eigenvalues[1], eigenvalues[2], true );
            }

            #define CalcEigenVectors33(i)                                                                \
            {                                                                                            \
                const Matrix33 M = A - Matrix33( eigenvalues[i], eigenvalues[i], eigenvalues[i] ); \
                const T &a=M._00, &b=M._01, &c=M._02;                                                    \
                const T &p=M._10, &q=M._11, &r=M._12;                                                    \
                T& x = eigenvectors[i].x;                                                                \
                T& y = eigenvectors[i].y;                                                                \
                T& z = eigenvectors[i].z = 1;                                                            \
                SolveSimultaneousEqn( a,b,c, p,q,r, x,y );                                               \
                eigenvectors[i].normalize();                                                             \
            }

            CalcEigenVectors33(0);
            CalcEigenVectors33(1);
            CalcEigenVectors33(2);

            return true;
        }

        void write( std::ofstream& fout ) const
        {
            fout.write( (char*)v, 9*sizeof(T) );
        }

        void read( std::ifstream& fin )
        {
            fin.read( (char*)v, 9*sizeof(T) );
        }

        bool save( const char* filePathName ) const
        {
            std::ofstream fout( filePathName, std::ios::out|std::ios::binary );

            if( fout.fail() )
            {
                COUT << "Error@Matrix33::save(): Failed to open file " << filePathName << ENDL;
                return false;
            }

            Matrix33::write( fout );

            return true;
        }

        bool load( const char* filePathName )
        {
            std::ifstream fin( filePathName, std::ios::out|std::ios::binary );

            if( fin.fail() )
            {
                COUT << "Error@Matrix33::save(): Failed to open file " << filePathName << ENDL;
                return false;
            }

            Matrix33::write( fin );

            return true;
        }
};

template <typename T>
BORA_FUNC_QUAL
inline Matrix33<T> operator*( const T s, const Matrix33<T>& m )
{
    Matrix33<T> tmp( m );
    for( int i=0; i<9; ++i ) { tmp.v[i] *= s; }
    return tmp;
}

template <typename T>
inline std::ostream& operator<<( std::ostream& os, const Matrix33<T>& m )
{
    std::string ret;
    std::string indent;

    const int indentation = 0;
    indent.append( indentation+1, ' ' );

    ret.append( "[" );

    for( int i=0; i<3; ++i )
    {
        ret.append( "[" );

        for( int j=0; j<3; ++j )
        {
            if( j ) { ret.append(", "); }
            ret.append( std::to_string( m(i,j) ) );
        }

        ret.append("]");

        if( i< 3-1 )
        {
            ret.append( ",\n" );
            ret.append( indent );
        }
    }

    ret.append( "]" );

    os << ret;

    return os;
}

typedef Matrix33<float>  Mat33f;
typedef Matrix33<double> Mat33d;

BORA_NAMESPACE_END

#endif

