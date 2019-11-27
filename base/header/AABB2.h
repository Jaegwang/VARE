//---------//
// AABB2.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.03.08                               //
//-------------------------------------------------------//

#ifndef _BoraAABB2_h_
#define _BoraAABB2_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

/// @brief A 2D axis-aligned bounding box.
/**
	A bounding box describes a volume in 2D world space that bounds a piece of geometry.
	The box is defined by two corner points which describe the minimum and maximum of the box.
*/
template <typename T>
class AABB2
{
    private:

        bool _initialized = false;

        Vector2<T> _min;
        Vector2<T> _max;

    public:

		BORA_FUNC_QUAL
        AABB2()
        {
			// nothing to do
		}

		BORA_FUNC_QUAL
        AABB2( const AABB2& a )
        {
            AABB2::operator=( a );
        }

		BORA_FUNC_QUAL
        AABB2( const Vector2<T>& p, const Vector2<T>& q )
        {
            AABB2::set( p, q );
        }

		BORA_FUNC_QUAL
        void reset()
        {
            _initialized = false;

            _min.zeroize();
            _max.zeroize();
        }

		BORA_FUNC_QUAL
        AABB2& operator=( const AABB2& a )
        {
            _initialized = a._initialized;

            _min = a._min;
            _max = a._max;

            return (*this);
        }

		BORA_FUNC_QUAL
        AABB2& set( const Vec2f& p, const Vec2f& q )
        {
            _initialized = true;

            GetMinMax( p.x, q.x, _min.x, _max.x );
            GetMinMax( p.y, q.y, _min.y, _max.y );

            return (*this);
        }

		BORA_FUNC_QUAL
        AABB2& merge( const AABB2& a, const AABB2& b )
        {
            if( a._initialized != b._initialized )
            {
                COUT << "Error@AABB2::merge(): Type mismatch." << ENDL;
                AABB2::reset();
                return (*this);
            }

            if( a._initialized )
            {
                const Vector2<T>& minPt = a.minPoint();
                const Vector2<T>& maxPt = a.maxPoint();

                _min.x = Min( _min.x, minPt.x );
                _min.y = Min( _min.y, minPt.y );

                _max.x = Max( _max.x, maxPt.x );
                _max.y = Max( _max.y, maxPt.y );
            }

            if( b._initialized )
            {
                const Vector2<T>& minPt = b.minPoint();
                const Vector2<T>& maxPt = b.maxPoint();

                _min.x = Min( _min.x, minPt.x );
                _min.y = Min( _min.y, minPt.y );

                _max.x = Max( _max.x, maxPt.x );
                _max.y = Max( _max.y, maxPt.y );
            }

            _initialized = a._initialized; // = b._initialized;

            return (*this);

        }

		BORA_FUNC_QUAL
        AABB2& expand( const Vector2<T>& p )
        {
            if( _initialized )
            {
                _min.x = Min( _min.x, p.x );
                _min.y = Min( _min.y, p.y );

                _max.x = Max( _max.x, p.x );
                _max.y = Max( _max.y, p.y );
            }
            else
            {
                _initialized = true;
                _min = _max = p;
            }

            return (*this);
        }

		BORA_FUNC_QUAL
        AABB2& expand( const AABB2& a )
        {
            if( a._initialized )
            {
                if( _initialized )
                {
                    _min.x = Min( _min.x, a._min.x );
                    _min.y = Min( _min.y, a._min.y );

                    _max.x = Max( _max.x, a._max.x );
                    _max.y = Max( _max.y, a._max.y );
                }
                else
                {
                    AABB2::operator=( a );
                }
            }

            return (*this);
        }

        BORA_FUNC_QUAL
        AABB2& expand( T epsilon=(T)EPSILON )
        {
            if( _initialized )
            {
                _min.x -= epsilon;
                _min.y -= epsilon;

                _max.x += epsilon;
                _max.y += epsilon;
            }

            return (*this);
        }

		BORA_FUNC_QUAL
        AABB2& move( const Vector2<T>& translation )
        {
            _min += translation;
            _max += translation;

            return (*this);
        }

		BORA_FUNC_QUAL
        AABB2& scale( const T factor, const Vector2<T>& pivot )
        {
            _min.x = factor * ( _min.x - pivot.x ) + pivot.x;
            _max.x = factor * ( _max.x - pivot.x ) + pivot.x;

            _min.y = factor * ( _min.y - pivot.y ) + pivot.y;
            _max.y = factor * ( _max.y - pivot.y ) + pivot.y;

            return (*this);
        }

		BORA_FUNC_QUAL
        const Vector2<T>& minPoint() const
        {
            return _min;
        }

		BORA_FUNC_QUAL
        const Vector2<T>& maxPoint() const
        {
            return _max;
        }

		BORA_FUNC_QUAL
        Vector2<T> center() const
        {
            return Center( _min, _max );
        }

		BORA_FUNC_QUAL
        T xWidth() const
        {
            return ( _max.x - _min.x );
        }

		BORA_FUNC_QUAL
        T yWidth() const
        {
            return ( _max.y - _min.y );
        }

		BORA_FUNC_QUAL
        T width( int dim ) const
        {
            switch( dim )
            {
                default:
                case 0: { return ( _max.x - _min.x ); }
                case 1: { return ( _max.y - _min.y ); }
            }
        }

		BORA_FUNC_QUAL
        bool contains( const Vector2<T>& p ) const
        {
            if( !_initialized ) { return false; }

            if( p.x < _min.x ) { return false; }
            if( p.y < _min.y ) { return false; }

            if( p.x > _max.x ) { return false; }
            if( p.y > _max.y ) { return false; }

            return true;
        }

		BORA_FUNC_QUAL
        bool intersects( const AABB2& a ) const
        {
            if( !_initialized || !a._initialized ) { return false; }

            if( _max.x < a._min.x ) { return false; }
            if( _max.y < a._min.y ) { return false; }

            if( _min.x > a._max.x ) { return false; }
            if( _min.y > a._max.y ) { return false; }

            return true;
        }

		BORA_FUNC_QUAL
        T maxWidth() const
        {
            return Max( (_max.x-_min.x), (_max.y-_min.y) );
        }

		BORA_FUNC_QUAL
        T minWidth() const
        {
            return Min( (_max.x-_min.x), (_max.y-_min.y) );
        }

		BORA_FUNC_QUAL
        T diagonalLength() const
        {
            return Sqrt( Pow2(_max.x-_min.x) + Pow2(_max.y-_min.y) );
        }

		BORA_FUNC_QUAL
        T area() const
        {
            return ( (_max.x-_min.x) * (_max.y-_min.y) );
        }

		BORA_FUNC_QUAL
        int maxDimension() const
        {
            const T xDim = _max.x - _min.x;
            const T yDim = _max.y - _min.y;

            return ( ( xDim > yDim ) ? 0 : 1 );
        }

		BORA_FUNC_QUAL
        bool initialized() const
        {
            return ( _initialized ? true : false );
        }

		BORA_FUNC_QUAL
        void split( AABB2& child1, AABB2& child2 )
        {
            const int d = maxDimension();
            T kDiv = (T)0.5 * ( _min[d] + _max[d] );

            Vector2<T> bMin1( _min ), bMax1( _max );
            bMax1[d] = kDiv;
            child1.set( bMin1, bMax1 );

            Vector2<T> bMin2( _min ), bMax2( _max );
            bMin2[d] = kDiv;
            child2.set( bMin2, bMax2 );
        }

		BORA_FUNC_QUAL
        T distanceFromOutside( const Vector3<T>& p ) const
        {
            const T dx = (p.x<_min.x) ? (_min.x-p.x) : ( (p.x>_max.x) ? (p.x-_max.x) : 0 );
            const T dy = (p.y<_min.y) ? (_min.y-p.y) : ( (p.y>_max.y) ? (p.y-_max.y) : 0 );

            return Sqrt( dx*dx + dy*dy );
        }

		BORA_FUNC_QUAL
        T squaredDistanceFromOutside( const Vector3<T>& p ) const
        {
            const T dx = (p.x<_min.x) ? (_min.x-p.x) : ( (p.x>_max.x) ? (p.x-_max.x) : 0 );
            const T dy = (p.y<_min.y) ? (_min.y-p.y) : ( (p.y>_max.y) ? (p.y-_max.y) : 0 );

            const T dist2 = Pow2(dx) + Pow2(dy);

            return dist2;
        }

        void write( std::ofstream& fout ) const
        {
            fout.write( (char*)&_initialized, sizeof(int) );
            _min.write( fout );
            _max.write( fout );
        }

        void read( std::ifstream& fin )
        {
            fin.read( (char*)&_initialized, sizeof(int) );
            _min.read( fin );
            _max.read( fin );
        }
};

template <typename T>
std::ostream& operator<<( std::ostream& os, const AABB2<T>& object )
{
	os << "<AABB2>" << std::endl;
    os << " domain: " << object.minPoint() << " ~ " << object.maxPoint() << std::endl;
    os << " size  : " << object.width(0) << " x " << object.width(1) << " x " << object.width(2) << std::endl;
    os << std::endl;
	return os;
}

typedef AABB2<float>  AABB2f;
typedef AABB2<double> AABB2d;

BORA_NAMESPACE_END

#endif

