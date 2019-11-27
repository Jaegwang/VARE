//-----------//
// Vector3.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2018.04.19                               //
//-------------------------------------------------------//

#ifndef _BoraVector3_h_
#define _BoraVector3_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

/// @brief A 3D vector
/**
	This class implements a 3D of various data types in Bora system.
*/
template <typename T>
class Vector3
{
	public:

		union
		{
			struct { T x, y, z; }; // position, vector
			struct { T u, v, w; }; // texture coordinates
			struct { T r, g, b; }; // color
			struct { T i, j, k; }; // index
			T values[3];
		};

	public:

        //////////////////
        // constructors //

		BORA_FUNC_QUAL
		Vector3()
		: x(0), y(0), z(0)
		{}

		BORA_FUNC_QUAL
		Vector3( const Vector3<T>& v )
		: x(v.x), y(v.y), z(v.z)
		{}

		BORA_FUNC_QUAL
		Vector3( T s )
		: x(s), y(s), z(s)
		{}

		BORA_FUNC_QUAL
		Vector3( T X, T Y, T Z )
		: x(X), y(Y), z(Z)
		{}

		BORA_FUNC_QUAL
		Vector3( const glm::tvec3<T>& v )
		: x(v.x), y(v.y), z(v.z)
		{}

		BORA_FUNC_QUAL
		Vector3( const glm::tvec4<T>& v )
		: x(v.x), y(v.y), z(v.z)
		{}

/*
		BORA_FUNC_QUAL
		Vector3( const openvdb::math::Vec3<T>& v )
		: x(v.x()), y(v.y()), z(v.z())
		{}

		BORA_FUNC_QUAL
		Vector3( const openvdb::math::Vec4<T>& v )
		: x(v.x()), y(v.y()), z(v.z())
		{}
*/
        BORA_FUNC_QUAL
        Vector3& set( T X, T Y, T Z )
        {
            x = X;
            y = Y;
            z = Z;

            return (*this);
        }

		BORA_FUNC_QUAL
		void zeroize()
		{
			x = y = z = 0;
		}

		BORA_FUNC_QUAL 
		Vector3& operator+=( const Vector3& v )
		{
			x+=v.x; y+=v.y; z+=v.z;
			return (*this);
		}

		BORA_FUNC_QUAL
		Vector3& operator-=( const Vector3& v )
		{
			x-=v.x; y-=v.y; z-=v.z;
			return (*this);
		}

		BORA_FUNC_QUAL
		Vector3& operator*=( const T s )
		{
			x*=s; y*=s; z*=s;
			return (*this);
		}

		BORA_FUNC_QUAL
		Vector3& operator/=( const T s )
		{
			x/=s; y/=s; z/=s;
			return *this;
		}		

		BORA_FUNC_QUAL
		Vector3 operator+( const Vector3& v ) const
		{
			return Vector3( x+v.x, y+v.y, z+v.z );
		}

		BORA_FUNC_QUAL
		Vector3 operator-( const Vector3& v ) const
		{
			return Vector3( x-v.x, y-v.y, z-v.z );
		}

		BORA_FUNC_QUAL
		Vector3 operator-() const
		{
			return Vector3( -x, -y, -z );
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
		operator glm::tvec3<T>() const
		{
			return glm::tvec3<T>( x, y, z );
		}

		BORA_FUNC_QUAL
		operator glm::tvec4<T>() const
		{
			return glm::tvec4<T>( x, y, z, (T)0 );
		}
/*
		BORA_FUNC_QUAL
		operator openvdb::math::Vec3<T>() const
		{
			return openvdb::math::Vec3<T>(x,y,z);
		}

		BORA_FUNC_QUAL
		operator openvdb::math::Vec4<T>() const
		{
			return openvdb::math::Vec4<T>(x,y,z,(T)0);
		}
*/
		BORA_FUNC_QUAL
		operator Vector3<float>() const
		{
			return Vector3<float>( (float)x, (float)y, (float)z );
		}

		BORA_FUNC_QUAL
		operator Vector3<double>() const
		{
			return Vector3<double>( (double)x, (double)y, (double)z );
		}		

		BORA_FUNC_QUAL
		Vector3& negate()
		{
			x=-x; y=-y, z=-z;
			return (*this);
		}

		BORA_FUNC_QUAL
		Vector3 negated() const
		{
			return Vector3( -x, -y, -z );
		}

		BORA_FUNC_QUAL
		Vector3& reverse()
		{
			x=-x; y=-y; z=-z;
			return (*this);
		}
		
		BORA_FUNC_QUAL
		Vector3 reversed() const
		{
			return Vector3( -x, -y, -z );
		}

        /////////////////////////
        // assignment operator //

		BORA_FUNC_QUAL
		Vector3& operator=( const Vector3& v )
		{
			x=v.x; y=v.y; z=v.z;
			return (*this);
		}

		BORA_FUNC_QUAL
		Vector3& operator=( const T s )
		{
			x = y = z = s;
			return (*this);
		}

        //////////////////////////
        // comparison operators //

        BORA_FUNC_QUAL
        bool equal( const Vector3& v, T epsilon=(T)1e-10 )
        {
            if( !AlmostSame( x, v.x, epsilon ) ) { return false; }
            if( !AlmostSame( y, v.y, epsilon ) ) { return false; }
            if( !AlmostSame( z, v.z, epsilon ) ) { return false; }
            return true;
        }

		BORA_FUNC_QUAL
		bool operator==( const Vector3& v ) const
		{
			if( x != v.x ) { return false; }
			if( y != v.y ) { return false; }
			if( z != v.z ) { return false; }
			return true;
		}

		BORA_FUNC_QUAL
		bool operator!=( const Vector3& v ) const
		{
			if( v.x != x ) { return true; }
			if( v.y != y ) { return true; }
			if( v.z != z ) { return true; }
			return false;
		}

		BORA_FUNC_QUAL
        bool operator<( const Vector3& v ) const
        {
            if( x >= v.x ) { return false; }
            if( y >= v.y ) { return false; }
            if( z >= v.z ) { return false; }
            return true;
        }

		BORA_FUNC_QUAL
        bool operator>( const Vector3& v ) const
        {
            if( x <= v.x ) { return false; }
            if( y <= v.y ) { return false; }
            if( z <= v.z ) { return false; }
            return true;
        }

		BORA_FUNC_QUAL
        bool operator<=( const Vector3& v ) const
        {
            if( x > v.x ) { return false; }
            if( y > v.y ) { return false; }
            if( z > v.z ) { return false; }
            return true;
        }

		BORA_FUNC_QUAL
        bool operator>=( const Vector3& v ) const
        {
            if( x < v.x ) { return false; }
            if( y < v.y ) { return false; }
            if( z < v.z ) { return false; }
            return true;
        }

		BORA_FUNC_QUAL
		T operator*( const Vector3& v ) const
		{
			return ( x*v.x + y*v.y + z*v.z );
		}

		BORA_FUNC_QUAL
		Vector3 operator*( const T s ) const 
		{
			return Vector3( x*s, y*s, z*s );
		}

		BORA_FUNC_QUAL
		Vector3 operator/( const T s ) const
		{
            const T d = 1.0 / ( s + EPSILON );
			return Vector3( x*d, y*d, z*d );
		}

		BORA_FUNC_QUAL
		Vector3 operator^( const Vector3& v ) const
		{
			return Vector3( y*v.z-z*v.y, z*v.x-x*v.z, x*v.y-y*v.x );
		}

		BORA_FUNC_QUAL
		T length() const
		{
			return Sqrt( x*x + y*y + z*z );
		}

		BORA_FUNC_QUAL
		T squaredLength() const
		{
			return ( x*x + y*y + z*z );
		}

		BORA_FUNC_QUAL
		Vector3& normalize()
		{
			const double d = 1 / Sqrt( x*x + y*y + z*z + EPSILON );
            x = (T)(x*d);
            y = (T)(y*d);
            z = (T)(z*d);
			return (*this);
		}

		BORA_FUNC_QUAL
		Vector3 normalized() const
		{
			const double d = 1 / Sqrt( x*x + y*y + z*z + EPSILON );
			return Vector3<T>( T(x*d), T(y*d), T(z*d) );
		}

		BORA_FUNC_QUAL
		Vector3 direction() const
		{
			const double d = 1 / Sqrt( x*x + y*y + z*z + EPSILON );
			return Vector3<T>( T(x*d), T(y*d), T(z*d) );
		}

		BORA_FUNC_QUAL
		T distanceTo( const Vector3& p ) const
		{
			return Sqrt( Pow2(x-p.x) + Pow2(y-p.y) + Pow2(z-p.z) );
		}

		BORA_FUNC_QUAL
		T squaredDistanceTo( const Vector3& p ) const
		{
			return ( Pow2(x-p.x) + Pow2(y-p.y) + Pow2(z-p.z) );
		}

        BORA_FUNC_QUAL
        Vector3& limitLength( const T targetLength )
        {
            const T lenSq = x*x + y*y + z*z;

            if( lenSq > Pow2(targetLength) )
            {
                const double d = (T)( targetLength / ( Sqrt(lenSq) + EPSILON ) );
                x = (T)(x*d);
                y = (T)(y*d);
                z = (T)(z*d);
            }

            return (*this);
        }

        BORA_FUNC_QUAL
		bool isFinite( const T infinite=(T)INFINITE )
		{
			if( Abs(x) > infinite ) { return false; }
			if( Abs(y) > infinite ) { return false; }
			if( Abs(z) > infinite ) { return false; }
			return true;			
		}

        BORA_FUNC_QUAL
        Vector3& cycle( bool toLeft )
        {
            if( toLeft )
            {
                const T x0 = x;
                x=y; y=z; z=x0;
            }
            else
            {
                const T z0 = z;
                z=y; y=x; x=z0;
            }
            return (*this);
        }

        BORA_FUNC_QUAL
        T indexOfMinElement() const
        {
            return ( (x<y) ? ( (x<z) ? 0 : 2 ) : ( (y<z) ? 1 : 2 ) );
        }

        BORA_FUNC_QUAL
        T indexOfMaxElement() const
        {
            return ( (x>y) ? ( (x>z) ? 0 : 2 ) : ( (y>z) ? 1 : 2 ) );
        }

        void write( std::ofstream& fout ) const
        {
            fout.write( (char*)&x, sizeof(T)*3 );
        }

        void read( std::ifstream& fin )
        {
            fin.read( (char*)&x, sizeof(T)*3 );
        }
};

template <typename T>
BORA_FUNC_QUAL
inline Vector3<T> operator*( const T s, const Vector3<T>& v )
{
    return Vector3<T>( s*v.x, s*v.y, s*v.z );
}

template <typename T>
inline std::ostream& operator<<( std::ostream& os, const Vector3<T>& v )
{
	os << "( " << v.x << ", " << v.y << ", " << v.z << " ) ";
	return os;
}

typedef Vector3<size_t> Idx3;
typedef Vector3<int>    Vec3i;
typedef Vector3<float>  Vec3f;
typedef Vector3<double> Vec3d;

/*
template <typename T>
inline void pyExport_Vector3()
{
	boost::python::class_< Vector3<T> >( "Vector3", boost::python::init<T,T,T>() )
		.def( "distanceTo", &Vector3<T>::distanceTo )
		.def( "squaredDistanceTo", &Vector3<T>::squaredDistanceTo )
		.def( boost::python::self + boost::python::self )
		.def( boost::python::self - boost::python::self );
}
*/
BORA_NAMESPACE_END

#endif

