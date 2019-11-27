//----------------------//
// VectorDenseField.cpp //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.11.15                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

VectorDenseField::VectorDenseField()
{
    // nothing to do
}

float
VectorDenseField::maxMagnitude( int& idx ) const
{
	int maxMagIdx = -1;
	float maxMag = -1.f;
	for( int i=0; i<numCells(); ++i )
	{
		const Vec3f vel = operator[](i);
		float mag = Sqrt(Pow2(vel.x)+Pow2(vel.y)+Pow2(vel.z));
		if( mag>maxMag ) { maxMag = mag; maxMagIdx = i; }
	}
	idx = maxMagIdx;
	return maxMag;
}

BORA_NAMESPACE_END

