//-----------//
// Vector4.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2018.09.10                               //
//-------------------------------------------------------//

#pragma once
#include <Bora.h>

BORA_NAMESPACE_BEGIN

/// @brief A 4D vector
/**
	This class implements a 4D of various data types in Bora system.
*/
template <typename T>
class Vector4
{
	public:

		union
		{
			struct { T x, y, z, w; }; // position
			struct { T r, g, b, a; }; // color
			struct { T i, j, k, l; }; // index
			T values[4];
		};

	public:

        //////////////////
        // constructors //

		BORA_FUNC_QUAL
		Vector4()
		: x(0), y(0), z(0), w(0)
		{}

		BORA_FUNC_QUAL
		Vector4( const Vector4<T>& v )
		: x(v.x), y(v.y), z(v.z), w(v.w)
		{}

		BORA_FUNC_QUAL
		Vector4( T s )
		: x(s), y(s), z(s), w(s)
		{}

		BORA_FUNC_QUAL
		Vector4( T X, T Y, T Z )
		: x(X), y(Y), z(Z), w(0)
		{}

		BORA_FUNC_QUAL
		Vector4( T X, T Y, T Z, T W )
		: x(X), y(Y), z(Z), w(W)
		{}

		BORA_FUNC_QUAL
		Vector4( const Vector3<T>& v, T W )
		: x(v.x), y(v.y), z(v.z), w(W)
		{}		

		BORA_FUNC_QUAL
		Vector4( const glm::tvec3<T>& v )
		: x(v.x), y(v.y), z(v.z), w((T)0)
		{}

		BORA_FUNC_QUAL
		Vector4( const glm::tvec4<T>& v )
		: x(v.x), y(v.y), z(v.z), w(v.w)
		{}
/*
		BORA_FUNC_QUAL
		Vector4( const openvdb::math::Vec3<T>& v )
		: x(v.x()), y(v.y()), z(v.z()), w((T)0)
		{}

		BORA_FUNC_QUAL
		Vector4( const openvdb::math::Vec4<T>& v )
		: x(v.x()), y(v.y()), z(v.z()), w(v.w())
		{}
*/
        BORA_FUNC_QUAL
        Vector4& set( T X, T Y, T Z, T W )
        {
            x = X;
            y = Y;
            z = Z;
            w = W;

            return (*this);
        }

		BORA_FUNC_QUAL
		void zeroize()
		{
			x = y = z = w = 0;
		}

		BORA_FUNC_QUAL 
		Vector4& operator+=( const Vector4& v )
		{
			x+=v.x; y+=v.y; z+=v.z; w+=v.w;
			return (*this);
		}

		BORA_FUNC_QUAL
		Vector4& operator-=( const Vector4& v )
		{
			x-=v.x; y-=v.y; z-=v.z; w-=v.w;
			return (*this);
		}

		BORA_FUNC_QUAL
		Vector4& operator*=( const T s )
		{
			x*=s; y*=s; z*=s; w*=s;
			return (*this);
		}

		BORA_FUNC_QUAL
		Vector4& operator/=( const T s )
		{
			x/=s; y/=s; z/=s; w/=s;
			return (*this);
		}		

		BORA_FUNC_QUAL
		Vector4 operator+( const Vector4& v ) const
		{
			return Vector4( x+v.x, y+v.y, z+v.z, w+v.w );
		}

		BORA_FUNC_QUAL
		Vector4 operator-( const Vector4& v ) const
		{
			return Vector4( x-v.x, y-v.y, z-v.z, w-v.w );
		}

		BORA_FUNC_QUAL
		Vector4 operator-() const
		{
			return Vector4( -x, -y, -z, -w );
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
			return glm::tvec3<T>(x,y,z);
		}

		BORA_FUNC_QUAL
		operator glm::tvec4<T>() const
		{
			return glm::tvec4<T>(x,y,z,w);
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
			return openvdb::math::Vec4<T>(x,y,z,w);
		}
*/
		BORA_FUNC_QUAL
		Vector4& negate()
		{
			x=-x; y=-y, z=-z, w=-w;
			return (*this);
		}

		BORA_FUNC_QUAL
		Vector4 negated() const
		{
			return Vector4( -x, -y, -z, -w );
		}

		BORA_FUNC_QUAL
		Vector4& reverse()
		{
			x=-x; y=-y; z=-z; w=-w;
			return (*this);
		}
		
		BORA_FUNC_QUAL
		Vector4 reversed() const
		{
			return Vector4( -x, -y, -z, -w );
		}

        /////////////////////////
        // assignment operator //

		BORA_FUNC_QUAL
		Vector4& operator=( const Vector4& v )
		{
			x=v.x; y=v.y; z=v.z; w=v.w;
			return (*this);
		}

		BORA_FUNC_QUAL
		Vector4& operator=( const T s )
		{
			x = y = z = w = s;
			return (*this);
		}

        //////////////////////////
        // comparison operators //

        BORA_FUNC_QUAL
        bool equal( const Vector4& v, T epsilon=(T)1e-10 )
        {
            if( !AlmostSame( x, v.x, epsilon ) ) { return false; }
            if( !AlmostSame( y, v.y, epsilon ) ) { return false; }
            if( !AlmostSame( z, v.z, epsilon ) ) { return false; }
            if( !AlmostSame( w, v.w, epsilon ) ) { return false; }
            return true;
        }

		BORA_FUNC_QUAL
		bool operator==( const Vector4& v ) const
		{
			if( x != v.x ) { return false; }
			if( y != v.y ) { return false; }
			if( z != v.z ) { return false; }
			if( w != v.w ) { return false; }
			return true;
		}

		BORA_FUNC_QUAL
		bool operator!=( const Vector4& v ) const
		{
			if( v.x != x ) { return true; }
			if( v.y != y ) { return true; }
			if( v.z != z ) { return true; }
			if( v.w != w ) { return true; }
			return false;
		}

		BORA_FUNC_QUAL
        bool operator<( const Vector4& v ) const
        {
            if( x >= v.x ) { return false; }
            if( y >= v.y ) { return false; }
            if( z >= v.z ) { return false; }
            if( w >= v.w ) { return false; }
            return true;
        }

		BORA_FUNC_QUAL
        bool operator>( const Vector4& v ) const
        {
            if( x <= v.x ) { return false; }
            if( y <= v.y ) { return false; }
            if( z <= v.z ) { return false; }
            if( w <= v.w ) { return false; }
            return true;
        }

		BORA_FUNC_QUAL
        bool operator<=( const Vector4& v ) const
        {
            if( x > v.x ) { return false; }
            if( y > v.y ) { return false; }
            if( z > v.z ) { return false; }
            if( w > v.w ) { return false; }
            return true;
        }

		BORA_FUNC_QUAL
        bool operator>=( const Vector4& v ) const
        {
            if( x < v.x ) { return false; }
            if( y < v.y ) { return false; }
            if( z < v.z ) { return false; }
            if( w < v.w ) { return false; }
            return true;
        }

		BORA_FUNC_QUAL
		T operator*( const Vector4& v ) const
		{
			return ( x*v.x + y*v.y + z*v.z + w*v.w );
		}

		BORA_FUNC_QUAL
		Vector4 operator*( const T v ) const 
		{
			return Vector4( x*v, y*v, z*v, w*v );
		}

		BORA_FUNC_QUAL
		Vector4 operator/( const T s ) const
		{
            const T d = 1 / ( s + EPSILON );
			return Vector4( x*d, y*d, z*d, w*d );
		}

		BORA_FUNC_QUAL
		T length() const
		{
			return Sqrt( x*x + y*y + z*z + w*w );
		}

		BORA_FUNC_QUAL
		T squaredLength() const
		{
			return ( x*x + y*y + z*z + w*w );
		}

		BORA_FUNC_QUAL
		Vector4& normalize()
		{
			const double d = 1 / Sqrt( x*x + y*y + z*z + w*w + EPSILON );
            x = (T)(x*d);
            y = (T)(y*d);
            z = (T)(z*d);
            w = (T)(w*d);
			return (*this);
		}

		BORA_FUNC_QUAL
		Vector4 normalized() const
		{
			const double d = 1 / Sqrt( x*x + y*y + z*z + w*w + EPSILON );
			return Vector4<T>( T(x*d), T(y*d), T(z*d), T(w*d) );
		}

		BORA_FUNC_QUAL
		Vector4 direction() const
		{
			const double d = 1 / Sqrt( x*x + y*y + z*z + w*w + EPSILON );
			return Vector4<T>( T(x*d), T(y*d), T(z*d), T(w*d) );
		}

		BORA_FUNC_QUAL
		T distanceTo( const Vector4& p ) const
		{
			return Sqrt( Pow2(x-p.x) + Pow2(y-p.y) + Pow2(z-p.z) + Pow2(w-p.w) );
		}

		BORA_FUNC_QUAL
		T squaredDistanceTo( const Vector4& p ) const
		{
			return ( Pow2(x-p.x) + Pow2(y-p.y) + Pow2(z-p.z) + Pow2(w-p.w) );
		}

        BORA_FUNC_QUAL
        Vector4& limitLength( const T targetLength )
        {
            const T lenSq = x*x + y*y + z*z + w*w;

            if( lenSq > Pow2(targetLength) )
            {
                const double d = targetLength / ( Sqrt(lenSq) + EPSILON );
                x = (T)(x*d);
                y = (T)(y*d);
                z = (T)(z*d);
                w = (T)(w*d);
            }

            return (*this);
        }

        BORA_FUNC_QUAL
		bool isFinite( const T infinite=(T)INFINITE )
		{
			if( Abs(x) > infinite ) { return false; }
			if( Abs(y) > infinite ) { return false; }
			if( Abs(z) > infinite ) { return false; }
			if( Abs(w) > infinite ) { return false; }
			return true;			
		}

        BORA_FUNC_QUAL
        Vector4& cycle( bool toLeft )
        {
            if( toLeft )
            {
                const T x0 = x;
                x=y; y=z; z=w; w=x0;
            }
            else
            {
                const T w0 = w;
                w=z; z=y; y=x; x=w0;
            }
            return (*this);
        }

        void write( std::ofstream& fout ) const
        {
            fout.write( (char*)&x, sizeof(T)*4 );
        }

        void read( std::ifstream& fin )
        {
            fin.read( (char*)&x, sizeof(T)*4 );
        }
};

template <typename T>
BORA_FUNC_QUAL
inline Vector4<T> operator*( const T s, const Vector4<T>& v )
{
    return Vector4<T>( s*v.x, s*v.y, s*v.z, s*v.w );
}

template <typename T>
inline std::ostream& operator<<( std::ostream& os, const Vector4<T>& v )
{
	os << "( " << v.x << ", " << v.y << ", " << v.z << ", " << v.w << " ) ";
	return os;
}

typedef Vector4<size_t> Idx4;
typedef Vector4<int>    Vec4i;
typedef Vector4<float>  Vec4f;
typedef Vector4<double> Vec4d;

/*
template <typename T>
inline void pyExport_Vector4()
{
	boost::python::class_< Vector4<T> >( "Vector4", boost::python::init<T,T,T,T>() )
		.def( "distanceTo", &Vector4<T>::distanceTo )
		.def( "squaredDistanceTo", &Vector4<T>::squaredDistanceTo )
		.def( boost::python::self + boost::python::self )
		.def( boost::python::self - boost::python::self );
}
*/
BORA_NAMESPACE_END

