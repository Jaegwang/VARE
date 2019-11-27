//-----------//
// Complex.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.03.08                               //
//-------------------------------------------------------//

#ifndef _BoraComplex_h_
#define _BoraComplex_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

template <typename T>
class Complex
{
    public:

        T r; // the real part
        T i; // the imaginary part

	public:

        BORA_FUNC_QUAL
		Complex()
        : r(0), i(0)
        {}

        BORA_FUNC_QUAL
		Complex( const Complex<T>& c )
        : r(c.r), i(c.i)
        {}

        BORA_FUNC_QUAL
		Complex( const T realPart, const T imaginaryPart=0 )
        : r(realPart), i(imaginaryPart)
        {}

        BORA_FUNC_QUAL
		Complex( const T a[2] )
        : r(a[0]), i(a[1])
        {}

        BORA_FUNC_QUAL
		Complex& setByPolar( const T& radius, const T& theta )
        {
            r = radius * Cos(theta);
            i = radius * Sin(theta);
            return (*this);
        }

        BORA_FUNC_QUAL
		void zeroize()
        {
            r = i = 0;
        }

        BORA_FUNC_QUAL
		Complex& operator=( const T realPart )
        {
            r = realPart;
            i = 0;
            return (*this);
        }

        BORA_FUNC_QUAL
		Complex& operator=( const Complex& c )
        {
            r = c.r;
            i = c.i;
            return (*this);
        }

        BORA_FUNC_QUAL
        bool operator==( const Complex& c ) const
        {
            if( r!=c.r ) { return false; }
            if( i!=c.i ) { return false; }
            return true;
        }

        BORA_FUNC_QUAL
        bool operator!=( const Complex& c ) const
        {
            if( r!=c.r ) { return true; }
            if( i!=c.i ) { return true; }
            return false;
        }

        BORA_FUNC_QUAL
        Complex& operator+=( const Complex& c )
        {
            r += c.r;
            i += c.i;
            return (*this);
        }

        BORA_FUNC_QUAL
        Complex& operator-=( const Complex& c )
        {
            r -= c.r;
            i -= c.i;
            return (*this);
        }

        BORA_FUNC_QUAL
        Complex& operator*=( const Complex& c )
        {
            const T r0 = r, i0 = i;
            r = (r0*c.r) - (i0*c.i);
            i = (r0*c.i) + (i0*c.r);
            return (*this);
        }

        BORA_FUNC_QUAL
        Complex& operator*=( const T s )
        {
            r *= s;
            i *= s;
            return (*this);
        }

        BORA_FUNC_QUAL
        Complex& operator/=( const T s )
        {
            const T _s = 1 / ( s + EPSILON );
            r *= _s;
            i *= _s;
            return (*this);
        }

        BORA_FUNC_QUAL
        Complex operator+( const Complex& c ) const
        {
            return Complex( r+c.r, i+c.i );
        }

        BORA_FUNC_QUAL
        Complex operator-( const Complex& c ) const
        {
            return Complex( r-c.r, i-c.i );
        }

        BORA_FUNC_QUAL
        Complex operator*( const Complex& c ) const
        {
            return Complex( r*c.r-i*c.i, r*c.i + i*c.r );
        }

        BORA_FUNC_QUAL
        Complex operator*( const T s ) const
        {
            return Complex( r*s, i*s );
        }

        BORA_FUNC_QUAL
        Complex operator/( const T s ) const
        {
            const T _s = 1 / ( s + EPSILON );
            return Complex( r*_s, i*_s );
        }

        BORA_FUNC_QUAL
		Complex operator-() const
        {
            return Complex( -r, -i );
        }

        BORA_FUNC_QUAL
		Complex& conjugate()
        {
            i = -i;
            return (*this);
        }

        BORA_FUNC_QUAL
		Complex conjugated() const
        {
            return Complex( r, -i );
        }

        BORA_FUNC_QUAL
		Complex& inverse()
        {
            const T _d = 1 / ( r*r + i*i + EPSILON );
            r *= _d;
            i *= _d;
            return (*this);
        }

        BORA_FUNC_QUAL
		Complex inversed() const
        {
            Complex c( *this );
            c.inverse();
            return c;
        }

        BORA_FUNC_QUAL
        bool isReal( const T tolerance=EPSILON )
        {
            if( AlmostZero(i,tolerance) ) { return true; }
            return false;
        }

        BORA_FUNC_QUAL
		T radius() const
        {
            return Sqrt( r*r + i*i );
        }

        BORA_FUNC_QUAL
		T squaredRadius() const
        {
            return ( r*r + i*i );
        }

        BORA_FUNC_QUAL
		T angle( bool zeroTo2Pi=true ) const
        {
            T ang = ATan2( i, r );
            if( zeroTo2Pi && ( ang<0 ) ) { ang += PI2; }
            return ang;
        }

        BORA_FUNC_QUAL
		Vector3<T> rotate( const Vector3<T>& p ) const
        {
            T R = r, I = i;
            T r2 = R*R+I*I;
            if( AlmostZero(r2) ) { return p; }
            if( !AlmostSame( r2, (T)1 ) )
            {
                T r = sqrtf(r2);
                R /= r;
                I /= r;
            }
            return Vector3<T>( R*p.x-I*p.y, I*p.x+R*p.y );
        }

        BORA_FUNC_QUAL
		Complex& rotate_counterClockwise_90()
        {
            Swap( r, i );
            r = -r;
            return (*this);
        }

        BORA_FUNC_QUAL
		Complex rotated_counterClockwise_90() const
        {
            return Complex( -i, r );
        }

        BORA_FUNC_QUAL
		Complex& rotate_clockwise_90()
        {
            Swap( r, i );
            i = -i;
            return (*this);
        }

        BORA_FUNC_QUAL
		Complex rotated_clockwise_90() const
        {
            return Complex( i, -r );
        }

        BORA_FUNC_QUAL
		static Complex plus_i()  // 0+i
        {
            return Complex( 0, 1 );
        }

        BORA_FUNC_QUAL
		static Complex minus_i() // 0-i
        {
            return Complex( 0, -1 );
        }

        BORA_FUNC_QUAL
		static Complex polar( const T& radius, const T& theta )
        {
            return Complex( radius*Cos(theta), radius*Sin(theta) );
        }

        BORA_FUNC_QUAL
		static Complex polar( const T& theta )
        {
            return Complex( Cos(theta), Sin(theta) );
        }

        BORA_FUNC_QUAL
		static Complex exp( const T& exponent )
        {
            return Complex( Cos(exponent), Sin(exponent) );
        }

		void write( std::ofstream& fout ) const
        {
            fout.write( (char*)&r, sizeof(T)*2 );
        }

		void read( std::ifstream& fin )
        {
            fin.read( (char*)&r, sizeof(T)*2 );
        }
};

template <typename T>
BORA_FUNC_QUAL
inline Complex<T> operator*( const T s, const Complex<T>& c )
{
    return Complex<T>( c.r*s, c.i*s );
}

template <typename T>
std::ostream& operator<<( std::ostream& os, const Complex<T>& c )
{
	os << "( " << c.r << " + " << c.i << "i )";
	return os;
}

typedef Complex<float>  Complexf;
typedef Complex<double> Complexd;

BORA_NAMESPACE_END

#endif

