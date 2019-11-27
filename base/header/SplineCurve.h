//---------------//
// SplineCurve.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2017.12.05                               //
//-------------------------------------------------------//

#pragma once
#include <Bora.h>

BORA_NAMESPACE_BEGIN

class SplineCurve
{
    public:

        PointArray cp;

    public:

        BORA_FUNC_QUAL const Vec3f begin() const { return cp[0]; }
        BORA_FUNC_QUAL const Vec3f end() const { return cp[cp.size()-1]; }

        BORA_FUNC_QUAL Vec3f point( double prm ) const
        {
			const int N = cp.length()-1;
			prm = Clamp( prm, 0.0, (double)N );

			const int i0 = Clamp( (int)(prm-1.0), 0, N );
			const int i1 = Clamp( (int)(prm)    , 0, N );
			const int i2 = Clamp( (int)(prm+1.0), 0, N );
			const int i3 = Clamp( (int)(prm+2.0), 0, N );

			const double t = prm - (double)i1;
			const double tt = t*t;
			const double ttt = t*t*t;

			float h00 = 2.0*ttt - 3.0*tt + 1;
			float h10 = ttt - 2.0*tt + t;
			float h01 = -2.0*ttt + 3.0*tt;
			float h11 = ttt - tt;

			const Vec3f& p_k0 = cp[i1];
			const Vec3f& p_k1 = cp[i2];

			const Vec3f m_k0 = ((cp[i1]+cp[i2])*0.5 - (cp[i0]+cp[i1])*0.5) * 0.5;
			const Vec3f m_k1 = ((cp[i2]+cp[i3])*0.5 - (cp[i1]+cp[i2])*0.5) * 0.5;

			return p_k0*h00 + m_k0*h10 + p_k1*h01 + m_k1*h11;
        }

};

BORA_NAMESPACE_END

