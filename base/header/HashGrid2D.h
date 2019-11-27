//--------------//
// HashGrid2D.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.08.20                               //
//-------------------------------------------------------//

#pragma once
#include <Bora.h>

BORA_NAMESPACE_BEGIN

class HashGrid2D
{
    private:

        std::unordered_map< int, std::vector<int> > _table;

        float _dv=1.f;
        int   _nx=1024;
        int   _ny=1024;
        int   _total=100000;

    public:

        void initialize( const Vec2f& min, const Vec2f& max, const int nx, const int ny )
        {
            _dv = Min( (max.x-min.x)/(float)nx, (max.y-min.y)/(float)ny );

            _nx = nx;
            _ny = ny;
        }

        void initialize( const Vec2f& min, const Vec2f& max, const float voxelSize )
        {
            _dv = voxelSize;

            _nx = (max.x-min.x)/voxelSize+1;
            _ny = (max.y-min.y)/voxelSize+1;
        }

        void clear()
        {
            for( auto it = _table.begin(); it != _table.end(); ++it )
            {
                it->second.clear();
            }
        }

        void indices( int& i, int& j, const float& x, const float& y )
        {
            i = x/_dv;
            j = y/_dv;
        }

        size_t hash( int i, int j ) const
        {
            i %= _nx;
            j %= _ny;

            if( i < 0 ) i += _nx;
            if( j < 0 ) j += _ny;

            return (j*_nx + i) % _total;            
        }

        size_t hash( const float x, const float y ) const 
        {
            int i = x/_dv;
            int j = y/_dv;

            return hash( i, j );
        }

        size_t hash( const Vec2f& p ) const 
        {
            return hash( p.x, p.y );
        }

        std::vector<int>& operator[] ( const int i )
        {
            return _table[i];
        }
};

BORA_NAMESPACE_END

