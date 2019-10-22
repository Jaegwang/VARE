#pragma once
#include <VARE.h>

VARE_NAMESPACE_BEGIN

template <typename T>
class Matrix22
{
    public:

        union
        {
            T v[4];               // four values
            Vector2<T> column[2]; // tow column vectors
            struct                // _ij
            {
                T _00, _10; // row[0]
                T _01, _11; // row[1]
            };
        };

    public:

        VARE_UNIFIED
        Matrix22()
        : v{ 1,0, 0,1 }
        {}

        VARE_UNIFIED
        Matrix22( const Matrix22& m )
        : v{ m._00,m._10, m._01,m._11 }
        {}

        VARE_UNIFIED
        Matrix22
        (
            const T m00, const T m01,
            const T m10, const T m11
        )
        : v{ m00,m10, m01,m11 }
        {}

        VARE_UNIFIED
        Matrix22( const T& s )
        : v{ s,s, s,s }
        {}

        VARE_UNIFIED
        Matrix22( const T a[4] )
        : v{ a[0],a[1], a[2],a[3] }
        {}

        VARE_UNIFIED
        Matrix22( const T x, const T y )
        : v{ x,0, 0,y }
        {}

        VARE_UNIFIED
        Matrix22( const Vector2<T>& c0, const Vector2<T>& c1 )
        {
            column[0] = c0;
            column[1] = c1;
        }
/*
        // OpenGL: column-major order
        VARE_UNIFIED
        Matrix22( const glm::tmat2x2<T>& m )
        : column{ m[0], m[1] }
        {}
*/
        VARE_UNIFIED
        Matrix22& operator=( const Matrix22& m )
        {
            for( int i=0; i<4; ++i ) { v[i] = m.v[i]; }
            return (*this);
        }

        VARE_UNIFIED
        Matrix22& set
        (
            const T m00, const T m01,
            const T m10, const T m11
        )
        {
            _00=m00; _01=m01;
            _10=m10; _11=m11;

            return (*this);
        }

        // the j-th column vector
        VARE_UNIFIED
        Vector2<T>& operator[]( const int j )
        {
            return column[j];
        }

        // the j-th column vector
        VARE_UNIFIED
        const Vector2<T>& operator[]( const int j ) const
        {
            return column[j];
        }

        // the (i,j)-element
        VARE_UNIFIED
        T& operator()( const int i, const int j )
        {
            return column[j][i];
        }

        // the (i,j)-element
        VARE_UNIFIED
        const T& operator()( const int i, const int j ) const
        {
            return column[j][i];
        }

        // the i-th row vector
        VARE_UNIFIED
        Vector2<T> row( const int i ) const
        {
            return Vector2<T>( column[0][i], column[1][i] );
        }

        VARE_UNIFIED
        bool operator==( const Matrix22& m )
        {
            for( int i=0; i<4; ++i )
            {
                if( v[i] != m.v[i] )
                {
                    return false;
                }
            }
            return true;
        }

        VARE_UNIFIED
        bool operator!=( const Matrix22& m )
        {
            for( int i=0; i<4; ++i )
            {
                if( v[i] != m.v[i] )
                {
                    return true;
                }
            }
            return false;
        }

        VARE_UNIFIED
        bool isSame( const Matrix22& m, const T tolerance=EPSILON ) const
        {
            for( int i=0; i<4; ++i )
            {
                if( !AlmostSame( v[i], m.v[i], tolerance ) )
                {
                    return false;
                }
            }

            return true;
        }

        VARE_UNIFIED
        bool isSymmetric( const T tolerance=EPSILON ) const
        {
            if( !AlmostSame( _01, _10, tolerance ) ) { return false; }

            return true;
        }

        VARE_UNIFIED
        bool isIdentity( const T tolerance=EPSILON ) const
        {
            for( int i=0; i<2; ++i )
            for( int j=0; j<2; ++j )
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

        VARE_UNIFIED
        Matrix22& operator+=( const Matrix22& m )
        {
            for( int i=0; i<4; ++i ) { v[i] += m.v[i]; }
            return (*this);
        }

        VARE_UNIFIED
        Matrix22& operator-=( const Matrix22& m )
        {
            for( int i=0; i<4; ++i ) { v[i] -= m.v[i]; }
            return (*this);
        }

        VARE_UNIFIED
        Matrix22& operator*=( const T s )
        {
            for( int i=0; i<4; ++i ) { v[i] *= s; }
            return *this;
        }

        VARE_UNIFIED
        Matrix22& operator*=( const Matrix22& m )
        {
            Matrix22 tmp( *this );

            for( int j=0; j<2; ++j )
            for( int i=0; i<2; ++i )
            {{
                T& d = column[j][i] = 0;

                for( int k=0; k<2; ++k )
                {
                    d += tmp.column[k][i] * m.column[j][k];
                }
            }}

            return (*this);
        }

        VARE_UNIFIED
        Matrix22 operator*( const T s ) const
        {
            return ( Matrix22(*this) *= s );
        }

        VARE_UNIFIED
        Matrix22 operator*( const Matrix22& m ) const
        {
            return ( Matrix22(*this) *= m );
        }

        VARE_UNIFIED
        Vector2<T> operator*( const Vector2<T>& v ) const
        {
            Vector2<T> tmp;
            tmp.x = _00 * v.x + _01 * v.y;
            tmp.y = _10 * v.x + _11 * v.y;
            return tmp;
        }

        VARE_UNIFIED
        Matrix22 operator+( const Matrix22& m ) const
        {
            Matrix22 tmp;
            for( int i=0; i<4; ++i ) { tmp.v[i] = v[i] + m.v[i]; }
            return tmp;
        }

        VARE_UNIFIED
        Matrix22 operator-( const Matrix22& m ) const
        {
            Matrix22 tmp;
            for( int i=0; i<4; ++i ) { tmp.v[i] = v[i] - m.v[i]; }
            return tmp;
        }
/*
        VARE_UNIFIED
        operator glm::tmat2x2<T>() const
        {
            return glm::tmat2x2<T>( column[0], column[1] );
        }        
*/
        VARE_UNIFIED
        Matrix22& transpose()
        {
            Swap(_01,_10);
            return (*this);
        }

        VARE_UNIFIED
        Matrix22 transposed() const
        {
            Matrix22 tmp( *this );
            tmp.transpose();
            return tmp;
        }

        VARE_UNIFIED
        T trace() const
        {
            return ( _00 + _11 );
        }

        VARE_UNIFIED
        double determinant() const
        {
            return double( _00*_11 - _01*_10 );
        }

        VARE_UNIFIED
        Matrix22& inverse()
        {
            const double _det = 1.0 / ( Matrix22::determinant() + EPSILON );

            const double m00=_00, m01=_01;
            const double m10=_10, m11=_11;

            _00 = (T)(  m11 * _det );
            _01 = (T)( -m01 * _det );

            _10 = (T)( -m10 * _det );
            _11 = (T)(  m00 * _det );

            return (*this);
        }

        VARE_UNIFIED
        Matrix22 inversed() const
        {
            return Matrix22(*this).inverse();
        }

        VARE_UNIFIED
        void identity()
        {
            _00=1; _01=0;
            _10=0; _11=1;
        }

        VARE_UNIFIED
        void setRotation( const T radians )
        {
            const T s = Sin( radians );
            const T c = Cos( radians );

            _00=c;   _01=-s;
            _10=s;   _11=c;
        }

        void eliminateScale()
        {
            column[0].normalize();
            column[1].normalize();
        }

        VARE_UNIFIED
        void addScale( const T sx, const T sy )
        {
            _00*=sx;  _01*=sy;
            _10*=sx;  _11*=sy;
        }

        void setScale( const T sx, const T sy )
        {
            Matrix22::eliminateScale();
            Matrix22::addScale( sx, sy );
        }

        void setScale( const Vector2<T>& s )
        {
            Matrix22::setScale( s.x, s.y );
        }

        void getScale( T& sx, T& sy ) const
        {
            sx = column[0].length();
            sy = column[1].length();
        }

        void getScale( Vector2<T>& s ) const
        {
            Matrix22::getScale( s.x, s.y );
        }

        void write( std::ofstream& fout ) const
        {
            fout.write( (char*)v, 4*sizeof(T) );
        }

        void decompose( Vector2<T>& r, Vector2<T>& s ) const
        {
            Matrix22 tmp( *this );

            tmp.getScale( s );
            tmp.eliminateScale();
            tmp.getRotation( r );
        }

        // Two eigenvectors are orthonormal each other only when the matrix is symmetric.
        bool eigen( T eigenvalues[2], Vector2<T> eigenvectors[2] ) const
        {
            const Matrix22& A = *this;

            // eigenvalues
            {
                const T a = 1;
                const T b = -A.trace();
                const T c = A.determinant();

                if( SolveQuadraticEqn( a,b,c, eigenvalues ) != 2 )
                {
                    COUT << "Error@Matrix33::eigen(): Invalid case." << ENDL;
                    return false;
                }

                if( eigenvalues[0] > eigenvalues[1] )
                {
                    Swap( eigenvalues[0], eigenvalues[1] );
                }
            }

            #define CalcEigenVectors22(i)                                                \
            {                                                                            \
                const Matrix22 M = A - Matrix22( eigenvalues[i], eigenvalues[i] ); \
                T& x = eigenvectors[i].x = 1;                                            \
                T& y = eigenvectors[i].y;                                                \
                if( Abs(M._01) > Abs(M._11) ) { y = -M._00*x / ( M._01 + EPSILON ); }    \
                else                          { y = -M._10*x / ( M._11 + EPSILON ); }    \
                eigenvectors[i].normalize();                                             \
            }

            CalcEigenVectors22(0);
            CalcEigenVectors22(1);

            return true;
        }

        void read( std::ifstream& fin )
        {
            fin.read( (char*)v, 4*sizeof(T) );
        }

        bool save( const char* filePathName ) const
        {
            std::ofstream fout( filePathName, std::ios::out|std::ios::binary );

            if( fout.fail() )
            {
                COUT << "Error@Matrix22::save(): Failed to open file " << filePathName << ENDL;
                return false;
            }

            Matrix22::write( fout );

            return true;
        }

        bool load( const char* filePathName )
        {
            std::ifstream fin( filePathName, std::ios::out|std::ios::binary );

            if( fin.fail() )
            {
                COUT << "Error@Matrix22::save(): Failed to open file " << filePathName << ENDL;
                return false;
            }

            Matrix22::write( fin );

            return true;
        }
};

template <typename T>
VARE_UNIFIED
inline Matrix22<T> operator*( const T s, const Matrix22<T>& m )
{
    Matrix22<T> tmp( m );
    for( int i=0; i<4; ++i ) { tmp.v[i] *= s; }
    return tmp;
}

template <typename T>
inline std::ostream& operator<<( std::ostream& os, const Matrix22<T>& m )
{
    std::string ret;
    std::string indent;

    const int indentation = 0;
    indent.append( indentation+1, ' ' );

    ret.append( "[" );

    for( int i=0; i<2; ++i )
    {
        ret.append( "[" );

        for( int j=0; j<2; ++j )
        {
            if( j ) { ret.append(", "); }
            ret.append( std::to_string( m(i,j) ) );
        }

        ret.append("]");

        if( i< 2-1 )
        {
            ret.append( ",\n" );
            ret.append( indent );
        }
    }

    ret.append( "]" );

    os << ret;

    return os;
}

typedef Matrix22<float>  Mat22f;
typedef Matrix22<double> Mat22d;

VARE_NAMESPACE_END
