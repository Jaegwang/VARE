
#pragma once
#include <VARE.h>

VARE_NAMESPACE_BEGIN

class SparseFrame
{
	private:

        int _fx, _fy, _fz;
        int _blockWidth=4;

        Grid _grid;
		MemorySpace _memType;

        Array<size_t> _pointers;
        Array<int>    _markers;
        Array<Idx3>   _coords;

        std::map< size_t, Grid* > _map;

		//
		int _fxfy;
		int _br, _brbr, _brbrbr;

    public:

        static SparseFrame* create( const Grid& vGrid, MemorySpace mType=kUnified );
        static void remove( SparseFrame* frame );

	public:

		const size_t nx() { return _fx; }
		const size_t ny() { return _fy; }
		const size_t nz() { return _fz; }

		const size_t blockWidth() { return _blockWidth; }

		const Grid& grid() { return _grid; }
		void registerGrid( Grid* g ) { _map[size_t(g)] = g; }

		size_t dataSize() { return _coords.size()*_brbrbr; }
		MemorySpace memorySpace() { return _memType; }

    public:

        void initialize( const Grid& vGrid, MemorySpace mType=kUnified );

        void build();
        void buildFromPoints( const PointArray& points, const bool enableTank=false, const float height=0.f );
		
		VARE_UNIFIED void indices( const size_t& n, size_t& i, size_t& j, size_t& k ) const
		{
			const Idx3& fCoord = _coords[n / _brbrbr];
			size_t bIdx = n % _brbrbr;

			i = fCoord.i*_br + (bIdx) % _br;
			j = fCoord.j*_br + (bIdx / _br) % _br;
			k = fCoord.k*_br + (bIdx / _brbr) % _br;
		}

		VARE_UNIFIED size_t index( const size_t i, const size_t j, const size_t k ) const
		{
			const size_t& p = _pointers[(k / _br)*_fxfy + (j / _br)*_fx + (i / _br)];
			if (p == INVALID_MAX) return p ;
			
			return (p + (k%_br)*_brbr + (j%_br)*_br + (i%_br));
		}
};

VARE_NAMESPACE_END

