//--------//
// Grid2D.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
//         Jaegwang Lim @ Dexter Studios                 //
//         Julie Jang @ Dexter Studios                   //
// last update: 2018.08.27                               //
//-------------------------------------------------------//

#pragma once
#include <Bora.h>

BORA_NAMESPACE_BEGIN

/// @brief A 2D uniform grid class.
/**
    This class implements a uniform grid in 2D or 3D space.

    We use three different spaces;
    1) voxel  space: (  0.0, 0.0 ) ~ (   Nx,   Ny )
    2) normal space: ( -0.5,-0.5 ) ~ ( +0.5, +0.5 )
    3) world  space: a space in which geometries reside

    All of the computations inside a Grid2D would be done in the voxel space.
    Therefore, world space geometry data must be transformed to the voxel space before being used in this class.

    We need four transformations;
    1) voxel  -> normal : to locate the grid initially in the world space
    2) normal -> world  : to transform the the grid by the parent xform node
    3) voxel  -> world  : to get the bounding box of the grid
    4) world  -> voxel  : to convert geometry data into the voxel space
*/
class Grid2D
{
    protected:

        // grid resolution
        size_t _nx; // Nx: # of cells in x-direction
        size_t _ny; // Ny: # of cells in y-direction

        // transformation matrices
        Mat33f _v2n; // voxel  -> normal
        Mat33f _n2w; // normal -> world (= the xform of the parent)
        Mat33f _v2w; // voxel  -> world (= _n2w * _v2n)
        Mat33f _w2v; // world  -> voxel (the inverse of _v2w)

    public:

        /////////////////
        // constructor //

        BORA_FUNC_QUAL
        Grid2D()
        {
            // initially, world space = normal space (i.e. _n2w is an identity matrix)
            Grid2D::initialize( 1, 1, AABB2f( Vec2f(-0.5f), Vec2f(0.5f) ) );
        }

        BORA_FUNC_QUAL
        Grid2D( const Grid2D& grid )
        {
            Grid2D::operator=( grid );
        }

        BORA_FUNC_QUAL
        Grid2D( const size_t Nx, const size_t Ny, const AABB2f& aabb )
        {
            Grid2D::initialize( Nx, Ny, aabb );
        }

        BORA_FUNC_QUAL
        Grid2D( const size_t subdivision, const AABB2f& aabb )
        {
            Grid2D::initialize( subdivision, aabb );
        }

        /////////////////////////
        // assignment operator //

        BORA_FUNC_QUAL
        Grid2D& operator=( const Grid2D& grid )
        {
            _nx = grid._nx;
            _ny = grid._ny;

            _v2n = grid._v2n;
            _n2w = grid._n2w;
            _v2w = grid._v2w;
            _w2v = grid._w2v;

            return (*this);
        }

        /////////////////
        // initializer //

        BORA_FUNC_QUAL
        bool initialize( const Grid2D& grid )
        {
            (*this) = grid;
            return true;
        }

        BORA_FUNC_QUAL
        bool initialize( const size_t Nx, const size_t Ny, const AABB2f& aabb )
        {
            if( Nx*Ny == 0 )
            {
                return false;
            }

            _nx = Nx;
            _ny = Ny;

            const float Lx = aabb.width(0);
            const float Ly = aabb.width(1);

            // voxel to normal
            _v2n = Mat33f( 1/(float)_nx, 1/(float)_ny, 1.f );
            _v2n.setTranslation( Vec2f(-0.5f) );

            // normal to world
            _n2w = Mat33f( Lx, Ly, 1.f );
            _n2w.setTranslation( aabb.center() );

            // voxel -> world = normal -> voxel -> world
            _v2w = _n2w * _v2n;

            // world to voxel (the inverse of _v2w)
            _w2v = _v2w.inversed();

            return true;
        }

        bool initialize( const size_t Nx, const size_t Ny, const Vec2f& min, const Vec2f& max )
        {
            if( Nx*Ny == 0 )
            {
                return false;
            }

            _nx = Nx;
            _ny = Ny;

            const float Lx = max.x - min.x;
            const float Ly = max.y - min.y;

            Vec2f center = (min+max)*0.5f;

            // voxel to normal
            _v2n = Mat33f( 1/(float)_nx, 1/(float)_ny, 1.f );
            _v2n.setTranslation( Vec2f(-0.5f) );

            // normal to world
            _n2w = Mat33f( Lx, Ly, 1.f );
            _n2w.setTranslation( center );

            // voxel -> world = normal -> voxel -> world
            _v2w = _n2w * _v2n;

            // world to voxel (the inverse of _v2w)
            _w2v = _v2w.inversed();

            return true;
        }

        bool initialize( const float voxelSize, const Vec2f& min, const Vec2f& max )
        {
            size_t x = (max.x - min.x) / voxelSize;
            size_t y = (max.y - min.y) / voxelSize;

            Vec2f inMin = min;
            Vec2f inMax = min + Vec2f( x*voxelSize, y*voxelSize );

            return initialize( x, y, inMin, inMax );
        }

        BORA_FUNC_QUAL
        bool initialize( const size_t subdivision, const AABB2f& aabb )
        {
            if( subdivision == 0 )
            {
                COUT << "Error@Grid2D::initialize(): Zero subdivision." << ENDL;
                return false;
            }

            const float Lx = aabb.width(0);
            const float Ly = aabb.width(1);

            const float maxDimension = Max( Lx, Ly );

            const float h = maxDimension / (float)subdivision;

            const size_t Nx = Round( Lx / h );
            const size_t Ny = Round( Lx / h );

            Grid2D::initialize( Nx, Ny, aabb );

            return true;
        }

        /////////////////////////
        // comparison operator //

        BORA_FUNC_QUAL
        bool operator==( const Grid2D& grid )
        {
            if( _nx != grid._nx ) { return false; }
            if( _ny != grid._ny ) { return false; }

            if( _v2n != grid._v2n ) { return false; }
            if( _n2w != grid._n2w ) { return false; }

            // We don't need compare _v2w and _w2v,
            // because they are derived from _v2n and _n2w.

            return true;
        }

        BORA_FUNC_QUAL
        bool operator!=( const Grid2D& grid )
        {
            if( _nx != grid._nx ) { return true; }
            if( _ny != grid._ny ) { return true; }

            if( _v2n != grid._v2n ) { return true; }
            if( _n2w != grid._n2w ) { return true; }

            // We don't need compare _v2w and _w2v,
            // because they are derived from _v2n and _n2w.

            return false;
        }

        //////////////////////
        // grid information //

        BORA_FUNC_QUAL
        size_t nx() const
        {
            return _nx;
        }

        BORA_FUNC_QUAL
        size_t ny() const
        {
            return _ny;
        }

        BORA_FUNC_QUAL
        size_t numCells() const
        {
            return ( _nx * _ny );
        }

        BORA_FUNC_QUAL
        size_t numVoxels() const
        {
            return ( _nx * _ny );
        }

        BORA_FUNC_QUAL
        AABB2f boundingBox() const
        {
            // the eight corner points of the grid in the voxel space
            const Vec2f p0( 0.f, 0.f );
            const Vec2f p1( _nx, 0.f );
            const Vec2f p2( 0.f, _ny );
            const Vec2f p3( _nx, _ny );

            // the eight corner points of the grid in the world space
            const Vec2f q0 = _v2w.transform( p0, false );
            const Vec2f q1 = _v2w.transform( p1, false );
            const Vec2f q2 = _v2w.transform( p2, false );
            const Vec2f q3 = _v2w.transform( p3, false );

            AABB2f aabb;

            aabb.expand( q0 );
            aabb.expand( q1 );
            aabb.expand( q2 );
            aabb.expand( q3 );

            return aabb;
        }

        std::string memorySize( const DataType dataType ) const
        {
            const size_t bytes = DataBytes( dataType ) * Grid2D::numCells();
            return MemorySize( bytes );
        }

        ////////////////////
        // index operator //

        BORA_FUNC_QUAL
        Vec2f cellCenter( const size_t i, const size_t j ) const
        {
            return Vec2f( i+.5f, j+.5f );
        }

        BORA_FUNC_QUAL
        Vec2f cellCenter( const Idx2& idx ) const
        {
            return cellCenter( idx.i, idx.j );
        }

        BORA_FUNC_QUAL
        size_t cellIndex( const size_t i, const size_t j ) const
        {
            return ( i + _nx*j );
        }

        BORA_FUNC_QUAL
        size_t cellIndex( const Idx2& ij ) const
        {
            return ( ij.i + _nx*ij.j );
        }

        BORA_FUNC_QUAL
        Idx2 cellIndices( const size_t idx ) const
        {
            return Idx2( (idx)%_nx, (idx/_nx)%_ny );
        }

        BORA_FUNC_QUAL
        void cellIndices( const size_t idx, size_t& i, size_t& j ) const
        {
            i = (idx)%_nx;
            j = (idx/_nx)%_ny;
        }

        BORA_FUNC_QUAL
        Idx2 cellIndices( const Vec2f& p /*voxel space*/ ) const
        {
            return Idx2( (size_t)p.x, (size_t)p.y );
        }

        BORA_FUNC_QUAL
        void getCellIndices( const Vec2f& p /*voxel space*/, size_t* i, size_t* j )
        {
            if( i ) { *i = (size_t)p.x; }
            if( j ) { *j = (size_t)p.y; }
        }

        BORA_FUNC_QUAL
        const Grid2D& getGrid2D() const { return *this; }        

        // the 1D array index of the west neighbor
        BORA_FUNC_QUAL
        size_t i0( const size_t& cellIndex ) const { return (cellIndex-1); }

        // the 1D array index of the east neighbor
        BORA_FUNC_QUAL
        size_t i1( const size_t& cellIndex ) const { return (cellIndex+1); }

        // the 1D array index of the south neighbor
        BORA_FUNC_QUAL
        size_t j0( const size_t& cellIndex ) const { return (cellIndex-_nx); }

        // the 1D array index of the north neighbor
        BORA_FUNC_QUAL
        size_t j1( const size_t& cellIndex ) const { return (cellIndex+_nx); }

        BORA_FUNC_QUAL
        void neighborCells( const Vec2f& point, Idx2& min, Idx2& max ) const
        {
            // Caution!!!
            /* Don't use size_t here.
               Below '_min' and '_max' could have nagitive values.
               That causes a overflow problem in which they are declared as size_t type.
            */
            Vec2i _min( (int)point.x-1, (int)point.y-1 );
            Vec2i _max( (int)point.x+1, (int)point.y+1 );

            min.i = Clamp( _min.i, 0, (int)_nx-1 );
            min.j = Clamp( _min.j, 0, (int)_ny-1 );

            max.i = Clamp( _max.i, 0, (int)_nx-1 );
            max.j = Clamp( _max.j, 0, (int)_ny-1 );
        }

        BORA_FUNC_QUAL
        void neighborCells( const Idx2& point, Idx2& min, Idx2& max ) const
        {
            // Caution!!!            
            /* Don't use size_t here.
               Below '_min' and '_max' could have nagitive values.
               That causes a overflow problem in which they are declared as size_t type.
            */

            Vec2i _min( (int)point.x-1, (int)point.y-1 );
            Vec2i _max( (int)point.x+1, (int)point.y+1 );

            min.i = Clamp( _min.i, 0, (int)_nx-1 );
            min.j = Clamp( _min.j, 0, (int)_ny-1 );

            max.i = Clamp( _max.i, 0, (int)_nx-1 );
            max.j = Clamp( _max.j, 0, (int)_ny-1 );
        }

        /////////////////////////
        // inside/outside test //

        BORA_FUNC_QUAL
        bool inside( const Vec2f& p /*voxel space*/, const float band=0.f ) const
        {
            if( p.x < band     ) { return false; }
            if( p.x > _nx-band ) { return false; }
            if( p.y < band     ) { return false; }
            if( p.y > _ny-band ) { return false; }
            return true;
        }

        BORA_FUNC_QUAL
        bool outside( const Vec2f& p /*voxel space*/, const float band=0.f ) const
        {
            if( p.x < band     ) { return true; }
            if( p.x > _nx-band ) { return true; }
            if( p.y < band     ) { return true; }
            if( p.y > _ny-band ) { return true; }
            return false;
        }

        ////////////////////
        // transformation //

        BORA_FUNC_QUAL
        Mat33f voxelToNormalMatrix() const
        {
            return _v2n;
        }

        BORA_FUNC_QUAL
        Mat33f normalToWorldMatrix() const
        {
            return _n2w;
        }

        BORA_FUNC_QUAL
        Mat33f voxelToWorldMatrix() const
        {
            return _v2w;
        }

        BORA_FUNC_QUAL
        Mat33f worldToVoxelMatrix() const
        {
            return _w2v;
        }

        BORA_FUNC_QUAL
        Vec2f worldPoint( const Vec2f& p /*voxel space*/ ) const
        {
            return ( _v2w.transform( p, false ) );
        }

        BORA_FUNC_QUAL
        Vec2f worldVector( const Vec2f& v /*voxel space*/ ) const
        {
            return ( _v2w.transform( v, true ) );
        }

        BORA_FUNC_QUAL
        Vec2f voxelPoint( const Vec2f& p /*world space*/ ) const
        {
            return ( _w2v.transform( p, false ) );
        }

        BORA_FUNC_QUAL
        Vec2f voxelVector( const Vec2f& v /*world space*/ ) const
        {
            return ( _w2v.transform( v, true ) );
        }

        BORA_FUNC_QUAL
        void applyTransform( const Mat33f& xform )
        {
            _n2w = xform;
        }

        ////////////
        // OpenGL //

        void glBeginNormalSpace( const bool toTranspose=false ) const
        {
            glPushMatrix();

            Mat44f m
            (
                _v2n._00, _v2n._01, 0.f, _v2n._02,
                _v2n._10, _v2n._11, 0.f, _v2n._12,
                _v2n._20, _v2n._21, 1.f,      0.f,
                     0.f,      0.f, 0.f,      1.f
            );

            if( toTranspose ) { m.transpose(); }

            glMultMatrixf( m.v );
        }

        void glEndNormalSpace() const
        {
            glPopMatrix();
        }

        void glBeginWorldSpace( const bool toTranspose=false ) const
        {
            glPushMatrix();

            Mat44f m
            (
                _v2w._00, _v2w._01, 0.f, _v2w._02,
                _v2w._10, _v2w._11, 0.f, _v2w._12,
                _v2w._20, _v2w._21, 1.f,      0.f,
                     0.f,      0.f, 0.f,      1.f
            );

            if( toTranspose ) { m.transpose(); }

            glMultMatrixf( m.v );
        }

        void glEndWorldSpace() const
        {
            glPopMatrix();
        }

        void draw() const
        {
            glBegin( GL_LINES );
            {
                for( size_t i=0; i<=_nx; ++i ) { glVertex2f(i,0.f); glVertex2f(i,_ny); }
                for( size_t j=0; j<=_ny; ++j ) { glVertex2f(0.f,j); glVertex2f(_nx,j); }
            }
            glEnd();
        }

        //////////
        // file //

        void write( std::ofstream& fout ) const
        {
            fout.write( (char*)&_nx, sizeof(size_t) );
            fout.write( (char*)&_ny, sizeof(size_t) );

            _v2n.write( fout );

            _v2w.write( fout );
        }

        void read( std::ifstream& fin )
        {
            fin.read( (char*)&_nx, sizeof(size_t) );
            fin.read( (char*)&_ny, sizeof(size_t) );

            _v2n.read( fin );

            _v2w.read( fin );
            _w2v = _v2w.inversed();
        }

        bool save( const char* filePathName ) const
        {
            std::ofstream fout( filePathName, std::ios::out|std::ios::binary );

            if( fout.fail() )
            {
                COUT << "Error@Grid2D::save(): Failed to open file " << filePathName << ENDL;
                return false;
            }

            Grid2D::write( fout );

            return true;
        }

        bool load( const char* filePathName )
        {
            std::ifstream fin( filePathName, std::ios::out|std::ios::binary );

            if( fin.fail() )
            {
                COUT << "Error@Grid2D::save(): Failed to open file " << filePathName << ENDL;
                return false;
            }

            Grid2D::read( fin );

            return true;
        }
};

BORA_NAMESPACE_END
