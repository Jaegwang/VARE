//-----------------//
// HashGridTable.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.08.20                               //
//-------------------------------------------------------//

#pragma once
#include <Bora.h>

BORA_NAMESPACE_BEGIN

class HashGridTable
{
    private:

        Idx2Array _cluster;

        float _dh=1.f;
        int   _nx=1024;
        int   _ny=1024;
        int   _nz=1024;

    private:

        BORA_FUNC_QUAL size_t hash( const int ni, const int nj, const int nk ) const
        {
            int i(ni), j(nj), k(nk);

            i %= _nx; j %= _ny; k %= _nz;
            if(i<0) i+=_nx;
            if(j<0) j+=_ny;
            if(k<0) k+=_nz;

            size_t t = i + j*_nx + k*_nx*_ny;
            return t % _cluster.size();
        }    

        BORA_FUNC_QUAL size_t hash( const Vec3f& p ) const
        {
            int i(p.x/_dh), j(p.y/_dh), k(p.z/_dh);
            return hash( i,j,k );
        }

    public:

        void initialize( const Grid& grid )
        {
            _nx = grid.nx();
            _ny = grid.ny();
            _nz = grid.nz();

            const size_t N = 256*256*256;
            _cluster.initialize( N, kUnified );
        }

        void initialize( const float dh, const size_t n )
        {
            _dh = dh;
            _cluster.initialize( n, kUnified );
        }

        void build( Vec3fArray& position, Vec3fArray& temp )
        {
            _cluster.zeroize();

            for( size_t n=0; n<position.size(); ++n )
            {
                const Vec3f& p = position[n];
                size_t t = hash( p );

                _cluster[t][0]++;
            }

            size_t count(0);
            for( size_t n=0; n<_cluster.size(); ++n )
            {
                size_t t = _cluster[n][0];

                if( t > 0 ) _cluster[n][0] = count;
                else        _cluster[n][0] = NULL_MAX;

                count += t;
            }

            for( size_t n=0; n < position.size(); ++n )
            {
                const Vec3f& p = position[n];
                size_t t = hash( p );

                size_t& idx = _cluster[t][0];
                size_t& num = _cluster[t][1];

                if( idx == NULL_MAX ) std::cout<<"NULL"<<std::endl;

                size_t loc = idx+num;

                temp[ loc ] = position[n];

                num++;
            }

            Vec3fArray::exchange( position, temp );
        }

        void build( const PointArray& position, IndexArray& sortIdx )
        {
            _cluster.zeroize();

            for( size_t n=0; n<position.size(); ++n )
            {
                const Vec3f& p = position[n];
                size_t t = hash( p );

                _cluster[t][0]++;
            }

            size_t count(0);
            for( size_t n=0; n<_cluster.size(); ++n )
            {
                size_t t = _cluster[n][0];

                if( t > 0 ) _cluster[n][0] = count;
                else        _cluster[n][0] = NULL_MAX;

                count += t;
            }

            sortIdx.initialize( position.size(), position.memorySpace() );

            for( size_t n=0; n < position.size(); ++n )
            {
                const Vec3f& p = position[n];
                size_t t = hash( p );

                size_t& idx = _cluster[t][0];
                size_t& num = _cluster[t][1];

                if( idx == NULL_MAX ) std::cout<<"NULL"<<std::endl;

                size_t loc = idx+num;

                sortIdx[ loc ] = n;

                num++;
            }
        }

        void build( Particles& pts )
        {   
            build( pts.position, pts.sortIdx );
        }

        BORA_FUNC_QUAL
        Idx2& operator()( const size_t i, const size_t j, const size_t k )
        {
            const size_t index = hash( i,j,k );
            return _cluster[ index ];
        }

        BORA_FUNC_QUAL
        const Idx2& operator()( const size_t i, const size_t j, const size_t k ) const
        {
            const size_t index = hash( i,j,k );
            return _cluster[ index ];
        }

        BORA_FUNC_QUAL
        Idx2& operator()( const Vec3f& p )
        {
            const size_t index = hash( p );
            return _cluster[ index ];
        }

        BORA_FUNC_QUAL
        const Idx2& operator()( const Vec3f& p ) const
        {
            const size_t index = hash( p );
            return _cluster[ index ];
        }

        BORA_FUNC_QUAL
        void getIndices( const Vec3f& p, int& i, int& j, int& k ) const
        {
            i = p.x/_dh; j = p.y/_dh; k = p.z/_dh;
        }

        BORA_FUNC_QUAL
        float getCellSize() const 
        {
            return _dh;
        }
        
};

BORA_NAMESPACE_END

