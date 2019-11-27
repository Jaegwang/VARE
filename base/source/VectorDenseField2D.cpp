//------------------------//
// VectorDenseField2D.cpp //
//------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
//         Julie Jang @ Dexter Studios                   //
// last update: 2018.04.25                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

VectorDenseField2D::VectorDenseField2D()
{
    // nothing to do
}

float
VectorDenseField2D::maxMagnitude( int& idx ) const
{
	int maxMagIdx = -1;
	float maxMag  = -1.f;
	for( int i=0; i<numCells(); ++i )
	{
		const Vec2f vel = operator[](i);
		float mag = Sqrt(Pow2(vel.x)+Pow2(vel.y));
		if( mag>maxMag ) { maxMag = mag; maxMagIdx = i; }
	}
	idx = maxMagIdx;
	return maxMag;
}

BORA_NAMESPACE_END

