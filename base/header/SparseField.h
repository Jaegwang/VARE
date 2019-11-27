//---------------//
// SparseField.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.10.04                               //
//-------------------------------------------------------//

#pragma once
#include <Bora.h>

BORA_NAMESPACE_BEGIN

template<class TT>
class SparseField : public Grid, public Array<TT>
{
    private:

        size_t _fx, _fy, _fz;
        size_t _fxy, _fxyz;
        size_t _br, _brr, _brrr;

        SparseFrame* _frame=0;

    public:

        TT background;

    public:

        void initialize( SparseFrame* inFrame )
        {
            _frame = (SparseFrame*)inFrame;

            _fx = _frame->_fx;
            _fy = _frame->_fy;
            _fz = _frame->_fz;
            
            _fxy = _fx*_fy;
            _fxyz = _fx*_fy*_fz;

            _br = _frame->_block_res;
            _brr = _br*_br;
            _brrr = _br*_br*_br;
     
            Grid::initialize( inFrame->_grid );

            inFrame->_map[size_t(this)] = (Grid*)this;
        }

        void initialize( SparseFrame* inFrame, const TT inBackground )
        {
            initialize( inFrame );
            background = (TT)inBackground;
        }

        BORA_UNIFIED
        const TT& operator() ( const size_t& i, const size_t& j, const size_t& k ) const
        {
            const size_t& p = _frame->_pointers[ (k/_br)*_fxy + (j/_br)*_fx + (i/_br) ];
            if( p == MAX_SIZE_T ) { return background; }
            return Array<TT>::operator[] ( p + (k%_br)*_brr + (j%_br)*_br + (i%_br) );
        }

        BORA_UNIFIED
        TT& operator() ( const size_t& i, const size_t& j, const size_t& k )
        {
            const size_t& p = _frame->_pointers[ (k/_br)*_fxy + (j/_br)*_fx + (i/_br) ];
            if( p == MAX_SIZE_T ) { 
                //std::cout<<"warning"<<std::endl;
                return background;                
            }
            return Array<TT>::operator[] ( p + (k%_br)*_brr + (j%_br)*_br + (i%_br) );
        }

        BORA_UNIFIED
        const TT& operator() ( const size_t& i, const size_t& j, const size_t& k, bool b ) const
        {
            return operator() (i,j,k);
        }

        BORA_UNIFIED
        const bool isWritable( const size_t& i, const size_t& j, const size_t& k ) const
        {
            const size_t& p = _frame->_pointers[ (k/_br)*_fxy + (j/_br)*_fx + (i/_br) ];
            if( p == MAX_SIZE_T ) return false;
            return true;
        }

        BORA_UNIFIED
        size_t findIndex( const size_t& i, const size_t& j, const size_t& k ) const 
        {
            const size_t& p = _frame->_pointers[ (k/_br)*_fxy + (j/_br)*_fx + (i/_br) ];
            if( p == MAX_SIZE_T ) return MAX_SIZE_T;
            return  p + (k%_br)*_brr + (j%_br)*_br + (i%_br);
        }

        BORA_UNIFIED
        size_t findIndex( const Idx3& idx ) const 
        {
            return findIndex( idx.i, idx.j, idx.k );
        }

        BORA_UNIFIED
        void findIndices( const size_t& n, size_t& i, size_t& j, size_t& k ) const
        {
            const Idx3& fCoord = _frame->_coords[ n / _brrr ];
            size_t bIdx = n % _brrr;

            i = fCoord.i*_br + (bIdx)%_br;
            j = fCoord.j*_br + (bIdx/_br)%_br;
            k = fCoord.k*_br + (bIdx/_brr)%_br;
        }

        BORA_UNIFIED
        Idx3 findIndices( const size_t& n ) const
        {
            size_t i,j,k;
            findIndices( n, i, j, k );
            return Idx3( i, j, k );
        }

        BORA_UNIFIED
        void neighborValues( TT values[6], const int i, const int j, const int k, const TT* def=0 )
        {
            const SparseField<TT>& f = (*this);
            const TT& v = f(i,j,k);

            if( def == 0 )
            {
                values[0] = i==0     ? 2.f*v-f(i+1,j,k) : f(i-1,j,k);
                values[1] = i==_nx-1 ? 2.f*v-f(i-1,j,k) : f(i+1,j,k);
                values[2] = j==0     ? 2.f*v-f(i,j+1,k) : f(i,j-1,k);
                values[3] = j==_ny-1 ? 2.f*v-f(i,j-1,k) : f(i,j+1,k);
                values[4] = k==0     ? 2.f*v-f(i,j,k+1) : f(i,j,k-1);
                values[5] = k==_nz-1 ? 2.f*v-f(i,j,k-1) : f(i,j,k+1);
            }
            else
            {
                values[0] = i==0     ? *def : f(i-1,j,k);
                values[1] = i==_nx-1 ? *def : f(i+1,j,k);
                values[2] = j==0     ? *def : f(i,j-1,k);
                values[3] = j==_ny-1 ? *def : f(i,j+1,k);
                values[4] = k==0     ? *def : f(i,j,k-1);
                values[5] = k==_nz-1 ? *def : f(i,j,k+1);
            }
            
        }

        BORA_UNIFIED
        TT lerp( const Vec3f& p ) const
        {
            const size_t i = Clamp( int(p.x-0.5f), 0, (int)_nx-2 );
            const size_t j = Clamp( int(p.y-0.5f), 0, (int)_ny-2 );
            const size_t k = Clamp( int(p.z-0.5f), 0, (int)_nz-2 );

            const Vec3f corner = cellCenter( i, j, k );

            float x = Clamp( (p.x - corner.x), 0.f, 1.f );
            float y = Clamp( (p.y - corner.y), 0.f, 1.f );
            float z = Clamp( (p.z - corner.z), 0.f, 1.f );

            const SparseField<TT>& f = (*this);

            TT c00 = f(i, j,   k  )*(1.f-x) + f(i+1, j,   k  )*x;
            TT c01 = f(i, j,   k+1)*(1.f-x) + f(i+1, j,   k+1)*x;
            TT c10 = f(i, j+1, k  )*(1.f-x) + f(i+1, j+1, k  )*x;
            TT c11 = f(i, j+1, k+1)*(1.f-x) + f(i+1, j+1, k+1)*x;

            TT c0 = c00*(1.f-y) + c10*y;
            TT c1 = c01*(1.f-y) + c11*y;

            return c0*(1.f-z) + c1*z;
        }

        BORA_UNIFIED
        TT lerpMin( const Vec3f& p ) const
        {
            const size_t i = Clamp( int(p.x-0.5f), 0, (int)_nx-2 );
            const size_t j = Clamp( int(p.y-0.5f), 0, (int)_ny-2 );
            const size_t k = Clamp( int(p.z-0.5f), 0, (int)_nz-2 );

            const Vec3f corner = cellCenter( i, j, k );

            float x = Clamp( (p.x - corner.x), 0.f, 1.f );
            float y = Clamp( (p.y - corner.y), 0.f, 1.f );
            float z = Clamp( (p.z - corner.z), 0.f, 1.f );

            const SparseField<TT>& f = (*this);

            TT v = Min( f(i, j, k), f(i+1, j,k ) );
            v = Min( v, Min( f(i, j  , k+1), f(i+1, j, k+1  ) ) );
            v = Min( v, Min( f(i, j+1, k  ), f(i+1, j+1, k  ) ) );
            v = Min( v, Min( f(i, j+1, k+1), f(i+1, j+1, k+1) ) );

            return v;
        }

        BORA_UNIFIED
        TT lerpMax( const Vec3f& p ) const
        {
            const size_t i = Clamp( int(p.x-0.5f), 0, (int)_nx-2 );
            const size_t j = Clamp( int(p.y-0.5f), 0, (int)_ny-2 );
            const size_t k = Clamp( int(p.z-0.5f), 0, (int)_nz-2 );

            const Vec3f corner = cellCenter( i, j, k );

            float x = Clamp( (p.x - corner.x), 0.f, 1.f );
            float y = Clamp( (p.y - corner.y), 0.f, 1.f );
            float z = Clamp( (p.z - corner.z), 0.f, 1.f );

            const SparseField<TT>& f = (*this);

            TT v = Max( f(i, j, k), f(i+1, j,k ) );
            v = Max( v, Max( f(i, j  , k+1), f(i+1, j, k+1  ) ) );
            v = Max( v, Max( f(i, j+1, k  ), f(i+1, j+1, k  ) ) );
            v = Max( v, Max( f(i, j+1, k+1), f(i+1, j+1, k+1) ) );

            return v;
        }        

        virtual bool build()
        {
            Array<TT>::initialize( _frame->_coords.size()*_brrr, _frame->_memType );
            Array<TT>::setValueAll( background );            
            return true;
        }
};

typedef SparseField<int>    IntSparseField;
typedef SparseField<float>  FloatSparseField;
typedef SparseField<float>  ScalarSparseField;
typedef SparseField<double> DoubleSparseField;
typedef SparseField<Vec2f>  Vec2fSparseField;
typedef SparseField<Vec3f>  Vec3fSparseField;
typedef SparseField<Vec3f>  VectorSparseField;
typedef SparseField<Vec4f>  Vec4fSparseField;
typedef SparseField<Vec2d>  Vec2dSparseField;
typedef SparseField<Vec3d>  Vec3dSparseField;
typedef SparseField<Vec4d>  Vec4dSparseField;
typedef SparseField<size_t> IdxSparseField;
typedef SparseField<size_t> IndexSparseField;

BORA_NAMESPACE_END

 