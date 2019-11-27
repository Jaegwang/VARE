//--------------//
// Quaternion.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2017.10.26                               //
//-------------------------------------------------------//

#ifndef _BoraQuaternion_h_
#define _BoraQuaternion_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

template <typename T>
class Quaternion
{
	public:

		union
		{
			struct { T w, x, y, z; };
			T values[4];
		};

    public:

        BORA_FUNC_QUAL
        Quaternion()
        : w(1), x(0), y(0), z(0)
        {}

        BORA_FUNC_QUAL
        Quaternion( const Quaternion<T>& q )
        : w(q.w), x(q.x), y(q.y), z(q.z)
        {}

        // from GML's float-precision quaternion
        BORA_FUNC_QUAL
		Quaternion( const glm::fquat& q )
		: w((T)q.w), x((T)q.x), y((T)q.y), z((T)q.z)
        {}

        // from GML's double-precision quaternion
        BORA_FUNC_QUAL
		Quaternion( const glm::dquat& q )
		: w((T)q.w), x((T)q.x), y((T)q.y), z((T)q.z)
        {}

        // to GML's float-precision quaternion
		BORA_FUNC_QUAL
		operator glm::fquat() const
		{
			return glm::fquat( (float)w, (float)x, (float)y, (float)z );
		}

        // to GML's double-precision quaternion
		BORA_FUNC_QUAL
		operator glm::dquat() const
		{
			return glm::fquat( (double)w, (double)x, (double)y, (double)z );
		}

		BORA_FUNC_QUAL
        T real() const
        {
            return w;
        }

		BORA_FUNC_QUAL
        Vector3<T> imaginaries() const
        {
            return Vector3<T>( x, y, z );
        }

        BORA_FUNC_QUAL
        Quaternion<T>& operator=( const Quaternion<T>& q )
        {
            w = q.w;
            x = q.x;
            y = q.y;
            z = q.z;

            return (*this);
        }

        void write( std::ofstream& fout ) const
        {
            fout.write( (char*)&w, sizeof(T)*4 );
        }

        void read( std::ifstream& fin )
        {
            fin.read( (char*)&w, sizeof(T)*4 );
        }
};

template <class T>
inline std::ostream& operator<<( std::ostream& os, const Quaternion<T>& q )
{
	COUT << "( " << q.w << " + " << q.x << "i + " << q.y << "j + " << q.z << "k )";
	return os;
}

typedef Quaternion<float>  Quatf;
typedef Quaternion<double> Quatd;

BORA_NAMESPACE_END

#endif

