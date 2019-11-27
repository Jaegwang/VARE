//----------------------//
// ScalarDenseField.cpp //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
//         Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.01.16                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

ScalarDenseField::ScalarDenseField()
{
    // nothing to do
}

float ScalarDenseField::min() const
{
//    return DenseField<float>::minValue();
    return 0.f;
}

float ScalarDenseField::max() const
{
//    return DenseField<float>::maxValue();
    return 0.f;
}

Vec3f ScalarDenseField::gradient( const size_t& i, const size_t& j, const size_t& k )
{
    ScalarDenseField& src = *this;
    const float& v = src(i,j,k);

    const float& v_i0 = i ==  0    ? v : src( i-1,j,k );
    const float& v_i1 = i == _nx-1 ? v : src( i+1,j,k );
    const float& v_j0 = j ==  0    ? v : src( i,j-1,k );
    const float& v_j1 = j == _ny-1 ? v : src( i,j+1,k );
    const float& v_k0 = k ==  0    ? v : src( i,j,k-1 );
    const float& v_k1 = k == _nz-1 ? v : src( i,j,k+1 );

    return Vec3f( (v_i1-v_i0)/0.5f, (v_j1-v_j0)/0.5f, (v_k1-v_k0)/0.5f );
}

Vec3f ScalarDenseField::gradient( const Idx3& idx )
{
    return gradient( idx.i, idx.j, idx.k );
}

BORA_NAMESPACE_END

