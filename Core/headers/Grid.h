
#pragma once
#include <VARE.h>

VARE_NAMESPACE_BEGIN

class Grid
{
	protected:

        // grid resolution
        size_t _nx; // Nx: # of cells in x-direction
        size_t _ny; // Ny: # of cells in y-direction
        size_t _nz; // Nz: # of cells in z-direction

        // sub-variables
        size_t _nxny; // = Nx * Ny (the stride for index k)

        Vec3f _translate;
        Vec3f _scale;

    public:

        /////////////////
        // constructor //
        VARE_UNIFIED
        Grid()
        {
            Grid::initialize( 1, 1, 1, AABB3f( Vec3f(-0.5f), Vec3f(0.5f) ) );
        }

        VARE_UNIFIED
        Grid( const Grid& grid )
        {
            Grid::operator=( grid );
        }

        VARE_UNIFIED
        Grid( const size_t Nx, const size_t Ny, const size_t Nz, const AABB3f& aabb )
        {
            Grid::initialize( Nx, Ny, Nz, aabb );
        }

        /////////////////////////
        // assignment operator //

        VARE_UNIFIED
        Grid& operator=( const Grid& grid )
        {
            _nx = grid._nx;
            _ny = grid._ny;
            _nz = grid._nz;
            _nxny = grid._nxny;

            _translate = grid._translate;
            _scale = grid._scale;

            return (*this);
        }

        /////////////////
        // initializer //

        VARE_UNIFIED bool initialize( const Grid& grid )
        {
            (*this) = grid;
            return true;
        }

        VARE_UNIFIED bool initialize( const size_t Nx, const size_t Ny, const size_t Nz, const AABB3f& aabb )
        {
            if( Nx*Ny*Nz == 0 ) return false;

            _nx = Nx;
            _ny = Ny;
            _nz = Nz;
            _nxny = Nx * Ny;

            const float Lx = aabb.width(0);
            const float Ly = aabb.width(1);
            const float Lz = ( Nz == 1 ) ? 1.f : aabb.width(2);

            _translate = aabb.minPoint();
            _scale = Vec3f( Lx/(float)_nx, Ly/(float)_ny, Lz/(float)_nz );

            return true;
        }

		VARE_UNIFIED bool initialize( const size_t Nx, const size_t Ny, const size_t Nz, const Vec3f& min, const Vec3f& max )
        {
            if( Nx*Ny*Nz == 0 ) return false;

            _nx = Nx;
            _ny = Ny;
            _nz = Nz;
            _nxny = Nx * Ny;

            const float Lx = max.x - min.x;
            const float Ly = max.y - min.y;
            const float Lz = ( Nz == 1 ) ? 1.f : max.z - min.z;

            _translate = min;
            _scale = Vec3f( Lx/(float)_nx, Ly/(float)_ny, Lz/(float)_nz );

            return true;
        }

		VARE_UNIFIED bool initialize( const Vec3f& gridCenter, const Vec3f& gridSize, const float voxelSize )
		{
			_nx = gridSize.x / voxelSize;
			_ny = gridSize.y / voxelSize;
			_nz = gridSize.z / voxelSize;
			_nxny = _nx*_ny;

			_translate = gridCenter - gridSize*0.5f;
			_scale = Vec3f( voxelSize );
		}

        /////////////////////////
        // comparison operator //

        VARE_UNIFIED
        bool operator==( const Grid& grid )
        {
            if( _nx != grid._nx ) { return false; }
            if( _ny != grid._ny ) { return false; }
            if( _nz != grid._nz ) { return false; }

            return true;
        }

        VARE_UNIFIED
        bool operator!=( const Grid& grid )
        {
            if( _nx != grid._nx ) { return true; }
            if( _ny != grid._ny ) { return true; }
            if( _nz != grid._nz ) { return true; }

            return false;
        }

        //////////////////////
        // grid information //

        VARE_UNIFIED
        bool is2D() const
        {
            return ( _nz == 1 );
        }

        VARE_UNIFIED
        bool is3D() const
        {
            return ( _nz > 1 );
        }

        VARE_UNIFIED
        size_t numCells() const
        {
            return ( _nx * _ny * _nz );
        }

        VARE_UNIFIED
        size_t numVoxels() const
        {
            return ( _nx * _ny * _nz );
        }

        VARE_UNIFIED
        AABB3f boundingBox() const
        {
            AABB3f aabb;

            aabb.expand( _translate );
            aabb.expand( _translate + Vec3f( (float)_nx*_scale.x,
                                             (float)_ny*_scale.y,
                                             (float)_nz*_scale.z ) );
            return aabb;
        }

        ////////////////////
        // index operator //

        VARE_UNIFIED
        Vec3f cellCenter( const size_t i, const size_t j, const size_t k=0 ) const
        {
            return Vec3f( i+.5f, j+.5f, k+.5f );
        }

        VARE_UNIFIED
        Vec3f cellCenter( const Idx3& idx ) const
        {
            return cellCenter( idx.i, idx.j, idx.k );
        }

        VARE_UNIFIED
        size_t cellIndex( const size_t i, const size_t j, const size_t k=0 ) const
        {
            return ( i + _nx*j + _nxny*k );
        }

        VARE_UNIFIED
        size_t cellIndex( const Idx3& ijk ) const
        {
            return ( ijk.i + _nx*ijk.j + _nxny*ijk.k );
        }

        VARE_UNIFIED
        Idx3 cellIndices( const size_t idx ) const
        {
            return Idx3( (idx)%_nx, (idx/_nx)%_ny, (idx/(_nx*_ny))%_nz );
        }

        VARE_UNIFIED
        void cellIndices( const size_t idx, size_t& i, size_t& j, size_t& k ) const
        {
            i = (idx)%_nx;
            j = (idx/_nx)%_ny;
            k = (idx/(_nx*_ny))%_nz;
        }

        VARE_UNIFIED
        Idx3 cellIndices( const Vec3f& p /*voxel space*/ ) const
        {
            return Idx3( (size_t)p.x, (size_t)p.y, (size_t)p.z );
        }

        VARE_UNIFIED
        void getCellIndices( const Vec3f& p /*voxel space*/, size_t* i, size_t* j, size_t* k=0 )
        {
            if( i ) { *i = (size_t)p.x; }
            if( j ) { *j = (size_t)p.y; }
            if( k ) { *k = (size_t)p.z; }
        }

        VARE_UNIFIED
        const Grid& getGrid() const { return *this; }        

        // the 1D array index of the west neighbor
        VARE_UNIFIED
        size_t i0( const size_t& cellIndex ) const { return (cellIndex-1); }

        // the 1D array index of the east neighbor
        VARE_UNIFIED
        size_t i1( const size_t& cellIndex ) const { return (cellIndex+1); }

        // the 1D array index of the south neighbor
        VARE_UNIFIED
        size_t j0( const size_t& cellIndex ) const { return (cellIndex-_nx); }

        // the 1D array index of the north neighbor
        VARE_UNIFIED
        size_t j1( const size_t& cellIndex ) const { return (cellIndex+_nx); }

        // the 1D array index of the back neighbor
        VARE_UNIFIED
        size_t k0( const size_t& cellIndex ) const { return (cellIndex-_nxny); }

        // the 1D array index of the forth neighbor
        VARE_UNIFIED
        size_t k1( const size_t& cellIndex ) const { return (cellIndex+_nxny); }

        VARE_UNIFIED
        void neighborCells( const Vec3f& point, Idx3& min, Idx3& max, const int offset=1 ) const
        {
            Vec3i _min( (int)point.x-offset, (int)point.y-offset, (int)point.z-offset );
            Vec3i _max( (int)point.x+offset, (int)point.y+offset, (int)point.z+offset );

            min.i = Clamp( _min.i, 0, (int)_nx-1 );
            min.j = Clamp( _min.j, 0, (int)_ny-1 );
            min.k = Clamp( _min.k, 0, (int)_nz-1 );

            max.i = Clamp( _max.i, 0, (int)_nx-1 );
            max.j = Clamp( _max.j, 0, (int)_ny-1 );
            max.k = Clamp( _max.k, 0, (int)_nz-1 );
        }

        VARE_UNIFIED
        void neighborCells( const Idx3& point, Idx3& min, Idx3& max, const int offset=1 ) const
        {
            Vec3i _min( (int)point.i-offset, (int)point.j-offset, (int)point.k-offset );
            Vec3i _max( (int)point.i+offset, (int)point.j+offset, (int)point.k+offset );

            min.i = Clamp( _min.i, 0, (int)_nx-1 );
            min.j = Clamp( _min.j, 0, (int)_ny-1 );
            min.k = Clamp( _min.k, 0, (int)_nz-1 );

            max.i = Clamp( _max.i, 0, (int)_nx-1 );
            max.j = Clamp( _max.j, 0, (int)_ny-1 );
            max.k = Clamp( _max.k, 0, (int)_nz-1 );
        }

		VARE_UNIFIED size_t nx() const { return _nx; }
		VARE_UNIFIED size_t ny() const { return _ny; }
		VARE_UNIFIED size_t nz() const { return _nz; }

        VARE_UNIFIED const float dx() const { return _scale.x; } 
        VARE_UNIFIED const float dy() const { return _scale.y; } 
        VARE_UNIFIED const float dz() const { return _scale.z; }

        VARE_UNIFIED const float lx() const { return _scale.x*(float)_nx; }
        VARE_UNIFIED const float ly() const { return _scale.y*(float)_ny; }
        VARE_UNIFIED const float lz() const { return _scale.z*(float)_nz; }

        VARE_UNIFIED const Vec3f minPoint() const { return _translate; }

        virtual bool build() { return true; }

        /////////////////////////
        // inside/outside test //

        VARE_UNIFIED
        bool inside( const Vec3f& p /*voxel space*/, const float band=0.f ) const
        {
            if( p.x < band            ) { return false; }
            if( p.x > (float)_nx-band ) { return false; }
            if( p.y < band            ) { return false; }
            if( p.y > (float)_ny-band ) { return false; }
            if( p.z < band            ) { return false; }
            if( p.z > (float)_nz-band ) { return false; }
            return true;
        }

        VARE_UNIFIED
        bool outside( const Vec3f& p /*voxel space*/, const float band=0.f ) const
        {
            if( p.x < band            ) { return true; }
            if( p.x > (float)_nx-band ) { return true; }
            if( p.y < band            ) { return true; }
            if( p.y > (float)_ny-band ) { return true; }
            if( p.z < band            ) { return true; }
            if( p.z > (float)_nz-band ) { return true; }
            return false;
        }

        VARE_UNIFIED
        Vec3f worldPoint( const Vec3f& p /*voxel space*/ ) const
        {
            return Vec3f( p.x*_scale.x + _translate.x,
                          p.y*_scale.y + _translate.y,
                          p.z*_scale.z + _translate.z );
        }

        VARE_UNIFIED
        Vec3f worldVector( const Vec3f& v /*voxel space*/ ) const
        {
            return Vec3f( v.x*_scale.x, v.y*_scale.y, v.z*_scale.z );
        }

        VARE_UNIFIED
        Vec3f voxelPoint( const Vec3f& p /*world space*/ ) const
        {
            return Vec3f( (p.x-_translate.x) / _scale.x,
                          (p.y-_translate.y) / _scale.y,
                          (p.z-_translate.z) / _scale.z );
        }

        VARE_UNIFIED
        Vec3f voxelVector( const Vec3f& v /*world space*/ ) const
        {
            return Vec3f( v.x/_scale.x, v.y/_scale.y, v.z/_scale.z );
        }
/*
        ////////////
        // OpenGL //

        void glBeginNormalSpace( const bool toTranspose=false ) const
        {
            glPushMatrix();
            glScalef( 1.f/(float)_nx, 1.f/(float)_ny, 1.f/(float)_nz );
        }

        void glEndNormalSpace() const
        {
            glPopMatrix();
        }

        void glBeginWorldSpace( const bool toTranspose=false ) const
        {
            glPushMatrix();
            glTranslatef( _translate.x, _translate.y, _translate.z );
            glScale( _scale.x, _scale.y, _scale.z );
        }

        void glEndWorldSpace() const
        {
            glPopMatrix();
        }

        void draw( bool x0, bool x1, bool y0, bool y1, bool z0, bool z1 ) const
        {
            if( x0 ) { _drawXSide( 0.f ); }
            if( x1 ) { _drawXSide( _nx ); }
            if( y0 ) { _drawYSide( 0.f ); }
            if( y1 ) { _drawYSide( _ny ); }
            if( z0 ) { _drawZSide( 0.f ); }
            if( z1 ) { _drawZSide( _nz ); }

            DrawCube( Vec3f(0,0,0), Vec3f(_nx,_ny,_nz) );
        }
*/
        //////////
        // file //

        void write( std::ofstream& fout ) const
        {
            fout.write( (char*)&_nx, sizeof(size_t) );
            fout.write( (char*)&_ny, sizeof(size_t) );
            fout.write( (char*)&_nz, sizeof(size_t) );
        }

        void read( std::ifstream& fin )
        {
            fin.read( (char*)&_nx, sizeof(size_t) );
            fin.read( (char*)&_ny, sizeof(size_t) );
            fin.read( (char*)&_nz, sizeof(size_t) );

            _nxny = _nx * _ny;
        }

        bool save( const char* filePathName ) const
        {
            std::ofstream fout( filePathName, std::ios::out|std::ios::binary );

            if( fout.fail() )
            {
                COUT << "Error@Grid::save(): Failed to open file " << filePathName << ENDL;
                return false;
            }

            Grid::write( fout );

            return true;
        }

        bool load( const char* filePathName )
        {
            std::ifstream fin( filePathName, std::ios::out|std::ios::binary );

            if( fin.fail() )
            {
                COUT << "Error@Grid::save(): Failed to open file " << filePathName << ENDL;
                return false;
            }

            Grid::read( fin );

            return true;
        }
/*
    protected:

        void _drawXSide( const float x ) const
        {
            glBegin( GL_LINES );
            {
                for( size_t j=0; j<=_ny; ++j ) { glVertex3f(x,j,0.f); glVertex3f(x,j,_nz); }
                for( size_t k=0; k<=_nz; ++k ) { glVertex3f(x,0.f,k); glVertex3f(x,_ny,k); }
            }
            glEnd();
        }

        void _drawYSide( const float y ) const
        {
            glBegin( GL_LINES );
            {
                for( size_t k=0; k<=_nz; ++k ) { glVertex3f(0.f,y,k); glVertex3f(_nx,y,k); }
                for( size_t i=0; i<=_nx; ++i ) { glVertex3f(i,y,0.f); glVertex3f(i,y,_nz); }
            }
            glEnd();
        }

        void _drawZSide( const float z ) const
        {
            glBegin( GL_LINES );
            {
                for( size_t i=0; i<=_nx; ++i ) { glVertex3f(i,0.f,z); glVertex3f(i,_ny,z); }
                for( size_t j=0; j<=_ny; ++j ) { glVertex3f(0.f,j,z); glVertex3f(_nx,j,z); }
            }
            glEnd();
        }
*/
};

VARE_NAMESPACE_END

