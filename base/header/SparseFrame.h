//---------------//
// SparseFrame.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2019.04.19                               //
//-------------------------------------------------------//

#pragma once

#include <Bora.h>

BORA_NAMESPACE_BEGIN

class SparseFrame
{
    public:

        int _fx, _fy, _fz;
        int _block_res=4;

        Grid _grid;
        MemorySpace _memType;

        Array<size_t> _pointers;
        Array<int>    _markers;
        Array<Idx3>   _coords;

        std::map< size_t, Grid* > _map;

    public:

        static SparseFrame* create( const Grid& vGrid, MemorySpace mType=kHost );
        static void remove( SparseFrame* frame );

    public:

        void initialize( const Grid& vGrid, MemorySpace mType=kHost );

        void build();
        void buildFromPoints( const PointArray& points, const bool enableTank=false, const float height=0.f );
};

BORA_NAMESPACE_END

