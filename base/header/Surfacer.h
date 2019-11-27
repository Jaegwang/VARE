//------------//
// Surfacer.h //
//-------------------------------------------------------//
// author: Julie Jang @ Dexter Digital                   //
// last update: 2018.04.10                               //
//-------------------------------------------------------//

#ifndef _BoraSurfacer_h_
#define _BoraSurfacer_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

class Surfacer
{
	protected:

		Vec3iArray		 _edg;
		ScalarDenseField _lvs;

	public:

		Surfacer();

		Surfacer( const Surfacer& source );

		void reset();

		Surfacer& operator=( const Surfacer& source );

        bool compute( const ScalarDenseField& lvs, TriangleMesh& mesh );
};


BORA_NAMESPACE_END

#endif

