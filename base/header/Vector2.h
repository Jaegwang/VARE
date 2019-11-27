//-----------//
// Vector2.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2018.04.25                               //
//-------------------------------------------------------//

#ifndef _BoraVector2_h_
#define _BoraVector2_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

/// @brief A 2D vector
/**
	This class implements a 2D of various data types in Bora system.
*/
template <typename T>
class Vector2
{
	public:

		union
		{
			struct { T x, y; }; // position, vector
			struct { T u, v; }; // texture coordinates
			struct { T s, t; }; // texture coordinates
			struct { T i, j; }; // index
			T values[2];
		};

	public:

        //////////////////
        // constructors //

		BORA_FUNC_QUAL
		Vector2()
		: x(0), y(0)
		{}

		BORA_FUNC_QUAL
		Vector2( const Vector2<T>& v )
		: x(v.x), y(v.y)
		{}

		BORA_FUNC_QUAL
		Vector2( T s )
		: x(s), y(s)
		{}

		BORA_FUNC_QUAL
		Vector2( T X, T Y )
		: x(X), y(Y)
		{}

		BORA_FUNC_QUAL
		Vector2( const glm::tvec2<T>& v )
		: x(v.x), y(v.y)
		{}

/*
		BORA_FUNC_QUAL
		Vector2( const openvdb::math::Vec2<T>& v )
		: x(v.x()), y(v.y())
		{}
*/
        BORA_FUNC_QUAL
        Vector2& set( T X, T Y )
        {
            x = X;
            y = Y;

            return (*this);
        }

		BORA_FUNC_QUAL
		void zeroize()
		{
			x = y = 0;
		}

		BORA_FUNC_QUAL 
		Vector2& operator+=( const Vector2& v )
		{
			x+=v.x; y+=v.y;
			return (*this);
		}

		BORA_FUNC_QUAL
		Vector2& operator-=( const Vector2& v )
		{
			x-=v.x; y-=v.y;
			return (*this);
		}

		BORA_FUNC_QUAL
		Vector2& operator*=( T s )
		{
			x*=s; y*=s;
			return (*this);
		}

		BORA_FUNC_QUAL
		Vector2& operator/=( T s )
		{
			x/=s; y/=s;
			return (*this);
		}		

		BORA_FUNC_QUAL
		Vector2 operator+( const Vector2& v ) const
		{
			return Vector2( x+v.x, y+v.y );
		}

		BORA_FUNC_QUAL
		Vector2 operator-( const Vector2& v ) const
		{
			return Vector2( x-v.x, y-v.y );
		}

		BORA_FUNC_QUAL
		Vector2 operator-() const
		{
			return Vector2( -x, -y );
		}

        //////////////////////
        // access operators //

		BORA_FUNC_QUAL
		T& operator[]( const int i )
		{
			return values[i];
		}

		BORA_FUNC_QUAL
		const T& operator[]( const int i ) const
		{
			return values[i];
		}

		BORA_FUNC_QUAL
		operator glm::tvec2<T>() const
		{
			return glm::tvec2<T>( x, y );
		}
/*
		BORA_FUNC_QUAL
		operator openvdb::math::Vec2<T>() const
		{
			return openvdb::math::Vec2<T>( x, y );
		}
*/
		BORA_FUNC_QUAL
		Vector2& negate()
		{
			x=-x; y=-y;
			return (*this);
		}

		BORA_FUNC_QUAL
		Vector2 negated() const
		{
			return Vector2( -x, -y );
		}

		BORA_FUNC_QUAL
		Vector2& reverse()
		{
			x=-x; y=-y;
			return (*this);
		}
		
		BORA_FUNC_QUAL
		Vector2 reversed() const
		{
			return Vector2( -x, -y );
		}

        /////////////////////////
        // assignment operator //

		BORA_FUNC_QUAL
		Vector2& operator=( const Vector2& v )
		{
			x=v.x; y=v.y;
			return (*this);
		}

		BORA_FUNC_QUAL
		Vector2& operator=( const T s )
		{
			x = y = s;
			return (*this);
		}

        //////////////////////////
        // comparison operators //

        BORA_FUNC_QUAL
        bool equal( const Vector2& v, T epsilon=(T)1e-10 )
        {
            if( !AlmostSame( x, v.x, epsilon ) ) { return false; }
            if( !AlmostSame( y, v.y, epsilon ) ) { return false; }
            return true;
        }

		BORA_FUNC_QUAL
		bool operator==( const Vector2& v ) const
		{
			if( x != v.x ) { return false; }
			if( y != v.y ) { return false; }
			return true;
		}

		BORA_FUNC_QUAL
		bool operator!=( const Vector2& v ) const
		{
			if( v.x != x ) { return true; }
			if( v.y != y ) { return true; }
			return false;
		}

		BORA_FUNC_QUAL
        bool operator<( const Vector2& v ) const
        {
            if( x >= v.x ) { return false; }
            if( y >= v.y ) { return false; }
            return true;
        }

		BORA_FUNC_QUAL
        bool operator>( const Vector2& v ) const
        {
            if( x <= v.x ) { return false; }
            if( y <= v.y ) { return false; }
            return true;
        }

		BORA_FUNC_QUAL
        bool operator<=( const Vector2& v ) const
        {
            if( x > v.x ) { return false; }
            if( y > v.y ) { return false; }
            return true;
        }

		BORA_FUNC_QUAL
        bool operator>=( const Vector2& v ) const
        {
            if( x < v.x ) { return false; }
            if( y < v.y ) { return false; }
            return true;
        }

		BORA_FUNC_QUAL
		T operator*( const Vector2& v ) const
		{
			return ( x*v.x + y*v.y );
		}

		BORA_FUNC_QUAL
		Vector2 operator*( const T s ) const 
		{
			return Vector2( x*s, y*s );
		}

		BORA_FUNC_QUAL
		Vector2 operator/( const T s ) const
		{
            const T d = 1.0 / ( s + EPSILON );
			return Vector2( x*d, y*d );
		}

		BORA_FUNC_QUAL
		T length() const
		{
			return Sqrt( x*x + y*y );
		}

		BORA_FUNC_QUAL
		T squaredLength() const
		{
			return ( x*x + y*y );
		}

		BORA_FUNC_QUAL
		Vector2& normalize()
		{
			const double d = 1 / Sqrt( x*x + y*y + EPSILON );
            x = (T)(x*d);
            y = (T)(y*d);
			return (*this);
		}

        BORA_FUNC_QUAL
        Vector2 normalized() const
        {
			const double d = 1 / Sqrt( x*x + y*y + EPSILON );
			return Vector2<T>( T(x*d), T(y*d) );
        }

		BORA_FUNC_QUAL
		Vector2 direction() const
		{
			const double d = 1 / Sqrt( x*x + y*y + EPSILON );
			return Vector2<T>( T(x*d), T(y*d) );
		}

		BORA_FUNC_QUAL
		T distanceTo( const Vector2& p ) const
		{
			return Sqrt( Pow2(x-p.x) + Pow2(y-p.y) );
		}

		BORA_FUNC_QUAL
		T squaredDistanceTo( const Vector2& p ) const
		{
			return ( Pow2(x-p.x) + Pow2(y-p.y) );
		}

        BORA_FUNC_QUAL
        Vector2& limitLength( const T targetLength )
        {
            const T lenSq = x*x + y*y;

            if( lenSq > Pow2(targetLength) )
            {
                const double d = (T)( targetLength / ( Sqrt(lenSq) + EPSILON ) );
                x = (T)(x*d);
                y = (T)(y*d);
            }

            return (*this);
        }

        BORA_FUNC_QUAL
		bool isFinite( const T infinite=(T)INFINITE )
		{
			if( Abs(x) > infinite ) { return false; }
			if( Abs(y) > infinite ) { return false; }
			return true;			
		}

        BORA_FUNC_QUAL
        T indexOfMinElement() const
        {
            return ( (x<y) ? 0 : 1 );
        }

        BORA_FUNC_QUAL
        T indexOfMaxElement() const
        {
            return ( (x>y) ? 0 : 1 );
        }

        void write( std::ofstream& fout ) const
        {
            fout.write( (char*)&x, sizeof(T)*2 );
        }

        void read( std::ifstream& fin )
        {
            fin.read( (char*)&x, sizeof(T)*2 );
        }
};

template <typename T>
BORA_FUNC_QUAL
inline Vector2<T> operator*( const T s, const Vector2<T>& v )
{
    return Vector2<T>( s*v.x, s*v.y );
}

template <typename T>
inline std::ostream& operator<<( std::ostream& os, const Vector2<T>& v )
{
	os << "( " << v.x << ", " << v.y << " ) ";
	return os;
}

typedef Vector2<size_t> Idx2;
typedef Vector2<int>    Vec2i;
typedef Vector2<float>  Vec2f;
typedef Vector2<double> Vec2d;

BORA_NAMESPACE_END

#endif

