//-------------------------//
// OceanTileVertexData.cpp //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.12.20                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

OceanTileVertexData::OceanTileVertexData()
{
    // nothing to do
}

void OceanTileVertexData::initialize( const OceanTile& oceanTile )
{
    if( oceanTile.initialized() == false ) { return; }

    _oceanTilePtr = const_cast<OceanTile*>( &oceanTile );

    const OceanParams& oceanParams = oceanTile.oceanParams();

    const int   N = oceanParams.resolution();
    const float L = oceanParams.geometricLength();

    OceanTileVertexData::buildMesh( N, L );

    _initialized = true;

    _toCalcStatistics = true;
}

void OceanTileVertexData::update( const float& time )
{
    if( _initialized == false ) { return; }

    const OceanTile& oceanTile = *_oceanTilePtr;
    const OceanParams& oceanParams = oceanTile.oceanParams();

    const int&   N = _N;
    const float& L = _L;
    const float& h = _h;

    OceanFunctor_Vertex F;
    {
        F.N = N;
        F.h = h;

        F.amplitudeGain = oceanParams.amplitudeGain * oceanParams.sceneConvertingScale;
        F.pinch         = oceanParams.pinch * oceanParams.sceneConvertingScale;

        F.crestGain         = oceanParams.crestGain;
        F.crestBias         = oceanParams.crestBias;
        F.crestAccumulation = oceanParams.crestAccumulation;
        F.crestDecay        = oceanParams.crestDecay;

        F.Dh = oceanTile.Dh.pointer();
        F.Dx = oceanTile.Dx.pointer();
        F.Dy = oceanTile.Dy.pointer();
        F.J  = oceanTile.J .pointer();

        F.GRD = GRD.pointer();
        F.POS = POS.pointer();
        F.NRM = NRM.pointer();
        F.WAV = WAV.pointer();
        F.CRS = CRS.pointer();
    }

    const size_t grainSize = oceanParams.grainSize;

    tbb::parallel_for( tbb::blocked_range2d<int>{ 0,N+1,1, 0,N+1,grainSize }, F );

    if( _toCalcStatistics )
    {
        minHeight =  INFINITE;
        maxHeight = -INFINITE;

        for( int i=0; i<(N+1)*(N+1); ++i )
        {
            const float y = POS[i].y;

            minHeight = Min( minHeight, y );
            maxHeight = Max( maxHeight, y );
        }

        minHeight *= 1.5f;
        maxHeight *= 1.5f;

        _toCalcStatistics = false;
    }

    // applying the flow speed
    if( oceanParams.flowSpeed > 1e-6f )
    {
        tmpPOS.swap( POS );
        tmpNRM.swap( NRM );
        tmpWAV.swap( WAV );
        tmpCRS.swap( CRS );

        OceanFunctor_Interpolation F;
        {
            F.N = N;
            F.L = L;
            F.h = h;

            F.eps = eps;
            F.time = time;
            F.flowSpeed = oceanParams.flowSpeed;
            F.wind.x = Cos( DegToRad( oceanParams.windDirection ) );
            F.wind.z = Sin( DegToRad( oceanParams.windDirection ) );

            F.tmpPOS = tmpPOS.pointer();
            F.tmpNRM = tmpNRM.pointer();
            F.tmpWAV = tmpWAV.pointer();
            F.tmpCRS = tmpCRS.pointer();

            F.POS = POS.pointer();
            F.NRM = NRM.pointer();
            F.WAV = WAV.pointer();
            F.CRS = CRS.pointer();
        }

        tbb::parallel_for( tbb::blocked_range2d<int>{ 0,N+1,1, 0,N+1,grainSize }, F );
    }
}

const OceanTile& OceanTileVertexData::oceanTile() const
{
    return ( *_oceanTilePtr );
}

bool OceanTileVertexData::initialized() const
{
    return _initialized;
}

int OceanTileVertexData::vertexIndex( const int& i, const int& j ) const
{
    return ( i + (_N+1)*j );
}

int OceanTileVertexData::resolution() const
{
    return _N;
}

float OceanTileVertexData::tileSize() const
{
    return _L;
}

float OceanTileVertexData::cellSize() const
{
    return _h;
}

void OceanTileVertexData::drawWireframe( bool normals, bool crests ) const
{
    if( TRI.size() == 0 ) { return; }

    glBegin( GL_LINES );
    {
        int index = 0;

        for( int j=0; j<_N; ++j )
        {
            for( int i=0; i<_N; ++i, index+=6 )
            {
                // the first triangle
                {
                    const int& v0 = TRI[index  ];
                    const int& v1 = TRI[index+1];
                    const int& v2 = TRI[index+2];

                    if( crests )
                    {
                        glColor( CRS[v0] ); glVertex( POS[v0] );
                        glColor( CRS[v1] ); glVertex( POS[v1] );

                        glColor( CRS[v1] ); glVertex( POS[v1] );
                        glColor( CRS[v2] ); glVertex( POS[v2] );

                        glColor( CRS[v2] ); glVertex( POS[v2] );
                        glColor( CRS[v0] ); glVertex( POS[v0] );
                    }
                    else
                    {
                        glVertex( POS[v0] );
                        glVertex( POS[v1] );

                        glVertex( POS[v1] );
                        glVertex( POS[v2] );

                        glVertex( POS[v2] );
                        glVertex( POS[v0] );
                    }
                }

                // the second triangle
                {
                    const int& v0 = TRI[index+3];
                    const int& v1 = TRI[index+4];
                    const int& v2 = TRI[index+5];

                    if( crests )
                    {
                        glColor( CRS[v0] ); glVertex( POS[v0] );
                        glColor( CRS[v1] ); glVertex( POS[v1] );

                        glColor( CRS[v1] ); glVertex( POS[v1] );
                        glColor( CRS[v2] ); glVertex( POS[v2] );

                        glColor( CRS[v2] ); glVertex( POS[v2] );
                        glColor( CRS[v0] ); glVertex( POS[v0] );
                    }
                    else
                    {
                        glVertex( POS[v0] );
                        glVertex( POS[v1] );

                        glVertex( POS[v1] );
                        glVertex( POS[v2] );

                        glVertex( POS[v2] );
                        glVertex( POS[v0] );
                    }
                }
            }
        }
    }
    glEnd();

    if( normals )
    {
        glColor3f( 1, 0, 0 );

        glBegin( GL_LINES );
        {
            const int numPoints = (int)POS.size();

            for( int i=0; i<numPoints; ++i )
            {
                const Vec3f& P = POS[i];
                const Vec3f& N = NRM[i];

                glVertex( P   );
                glVertex( P+N );
            }
        }
        glEnd();
    }
}

void OceanTileVertexData::drawOutline() const
{
    const Vec3f p0( 0.f, 0.f, 0.f );
    const Vec3f p1(  _L, 0.f, 0.f );
    const Vec3f p2(  _L, 0.f,  _L );
    const Vec3f p3( 0.f, 0.f,  _L );

    glBegin( GL_LINES );
    {
        glVertex( p0 );   glVertex( p1 );
        glVertex( p1 );   glVertex( p2 );
        glVertex( p2 );   glVertex( p3 );
        glVertex( p3 );   glVertex( p0 );
    }
    glEnd();
}

bool OceanTileVertexData::exportToEXR( const char* filePathName ) const
{
    if( FileExtension( filePathName ) != "exr" )
    {
        COUT << "Error@OceanTileVertexData::exportToEXR(): Invalid file extension." << ENDL;
        return false;
    }

    const int N = _N;

    Image img;

    img.create( N, N );

    for( int j=0; j<N; ++j )
    {
        const int stride = (N+1) * j;

        for( int i=0; i<N; ++i )
        {
            Pixel& p = img( i, j );

            const int index = i + stride;

            const Vec3f& P = POS[index];
            const Vec3f& G = GRD[index];

            p.r = P.x - G.x;
            p.g = P.y - G.y;
            p.b = P.z - G.z;
            p.a = CRS[index];
        }
    }

    return img.save( filePathName );
}

bool OceanTileVertexData::importFromEXR( const char* filePathName, const float L )
{
    if( FileExtension( filePathName ) != "exr" )
    {
        COUT << "Error@OceanTileVertexData::importFromEXR(): Invalid file extension." << ENDL;
        return false;
    }

    Image img;

    if( img.load( filePathName ) == false )
    {
        return false;
    }

    const int w = img.width();
    const int h = img.height();
    const int N = w;

    if( ( w != h ) || !IsPowersOfTwo(N) )
    {
        COUT << "Error@OceanTileVertexData::importFromEXR(): Invalid resolution." << ENDL;
        return false;
    }

    OceanTileVertexData::buildMesh( N, L );

    for( int j=0; j<N; ++j )
    {
        const int stride = (N+1) * j;

        for( int i=0; i<N; ++i )
        {
            Pixel& p = img( i, j );

            const int index = i + stride;

            Vec3f& P = POS[index];
            const Vec3f& G = GRD[index];

            P.r = G.x + p.r;
            P.g = G.y + p.g;
            P.b = G.z + p.b;

            CRS[index] = p.a;
        }
    }

    // the end lines
    for( int i=0; i<(N+1); ++i )
    {
        const int from = OceanTileVertexData::vertexIndex( i, 0 );
        const int to   = OceanTileVertexData::vertexIndex( i, N );

        GRD[to] = GRD[from];
        POS[to] = POS[from];
        CRS[to] = CRS[from];
    }

    // the end lines
    for( int j=0; j<(N+1); ++j )
    {
        const int from = OceanTileVertexData::vertexIndex( 0, j );
        const int to   = OceanTileVertexData::vertexIndex( N, j );

        GRD[to] = GRD[from];
        POS[to] = POS[from];
        CRS[to] = CRS[from];
    }

    // the corner points
    {
        const int from = OceanTileVertexData::vertexIndex( 0, 0 );
        const int to   = OceanTileVertexData::vertexIndex( N, N );

        GRD[to] = GRD[from];
        POS[to] = POS[from];
        CRS[to] = CRS[from];
    }

    return true;
}

void OceanTileVertexData::lerp( const Vec3f& worldPosition, Vec3f* grd, Vec3f* pos, Vec3f* nrm, Vec3f* wav, float* crs ) const
{
    const int&   N = _N;
    const float& L = _L;
    const float& h = _h;

    // (I, J): ocean tile index
    const int I = Floor( worldPosition.x / L );
    const int J = Floor( worldPosition.z / L );

    // it's origin in world space
    const Vec3f O( I*L, 0.f, J*L );

    // local position
    const float x = worldPosition.x - O.x;
    const float z = worldPosition.z - O.z;

    // cell indices
    const int i = Clamp( int(x/h), 0, N-1 );
    const int j = Clamp( int(z/h), 0, N-1 );

    const float a = x - (i*h);
    const float b = h - a;
    const float c = z - (j*h);
    const float d = h - c;

    const float hh = h*h;

    const float w0 = ( b * d ) / hh;
    const float w1 = ( a * d ) / hh;
    const float w2 = ( b * c ) / hh;
    const float w3 = ( a * c ) / hh;

    const int v0 = OceanTileVertexData::vertexIndex( i  , j   );
    const int v1 = OceanTileVertexData::vertexIndex( i+1, j   );
    const int v2 = OceanTileVertexData::vertexIndex( i  , j+1 );
    const int v3 = OceanTileVertexData::vertexIndex( i+1, j+1 );

    if( grd ) { *grd = w0*GRD[v0] + w1*GRD[v1] + w2*GRD[v2] + w3*GRD[v3] + O; }
    if( pos ) { *pos = w0*POS[v0] + w1*POS[v1] + w2*POS[v2] + w3*POS[v3] + O; }
    if( nrm ) { *nrm = w0*NRM[v0] + w1*NRM[v1] + w2*NRM[v2] + w3*NRM[v3];     }
    if( wav ) { *wav = w0*WAV[v0] + w1*WAV[v1] + w2*WAV[v2] + w3*WAV[v3];     }
    if( crs ) { *crs = w0*CRS[v0] + w1*CRS[v1] + w2*CRS[v2] + w3*CRS[v3];     }
}

void OceanTileVertexData::catrom( const Vec3f& worldPosition, Vec3f* grd, Vec3f* pos, Vec3f* nrm, Vec3f* wav, float* crs ) const
{
    const int&   N = _N;
    const float& L = _L;
    const float& h = _h;

    // (I, J): ocean tile index
    const int I = Floor( worldPosition.x / L );
    const int J = Floor( worldPosition.z / L );

    // it's origin in world space
    const Vec3f O( I*L, 0.f, J*L );

    // local position
    const float x = worldPosition.x - O.x;
    const float z = worldPosition.z - O.z;

    // cell indices
    const int i = Clamp( int(x/h), 0, N-1 );
    const int j = Clamp( int(z/h), 0, N-1 );

    const int i0 = Wrap( i-1, N );
    const int j0 = Wrap( j-1, N );

    const int i1 = i;
    const int j1 = j;

    const int i2 = Wrap( i+1, N );
    const int j2 = Wrap( j+1, N );

    const int i3 = Wrap( i+2, N );
    const int j3 = Wrap( j+2, N );

    const int idx00 = vertexIndex(i0,j0), idx10 = vertexIndex(i1,j0), idx20 = vertexIndex(i2,j0), idx30 = vertexIndex(i3,j0);
    const int idx01 = vertexIndex(i0,j1), idx11 = vertexIndex(i1,j1), idx21 = vertexIndex(i2,j1), idx31 = vertexIndex(i3,j1);
    const int idx02 = vertexIndex(i0,j2), idx12 = vertexIndex(i1,j2), idx22 = vertexIndex(i2,j2), idx32 = vertexIndex(i3,j2);
    const int idx03 = vertexIndex(i0,j3), idx13 = vertexIndex(i1,j3), idx23 = vertexIndex(i2,j3), idx33 = vertexIndex(i3,j3);

    const float fx = ( x - (i*h) ) / h;
    const float fz = ( z - (j*h) ) / h;

    Vec3f X0, X2, X3;
    if( i0 > i1 ) { X0.x = -L; }
    if( i2 < i1 ) { X2.x =  L; }
    if( i3 < i1 ) { X3.x =  L; }

    Vec3f Z0, Z2, Z3;
    if( j0 > j1 ) { Z0.z = -L; }
    if( j2 < j1 ) { Z2.z =  L; }
    if( j3 < j1 ) { Z3.z =  L; }

    if( grd )
    {
        const Vec3f V0 = CatRom( GRD[idx00]+X0, GRD[idx10], GRD[idx20]+X2, GRD[idx30]+X3, fx );
        const Vec3f V1 = CatRom( GRD[idx01]+X0, GRD[idx11], GRD[idx21]+X2, GRD[idx31]+X3, fx );
        const Vec3f V2 = CatRom( GRD[idx02]+X0, GRD[idx12], GRD[idx22]+X2, GRD[idx32]+X3, fx );
        const Vec3f V3 = CatRom( GRD[idx03]+X0, GRD[idx13], GRD[idx23]+X2, GRD[idx33]+X3, fx );

        *grd = CatRom( V0+Z0, V1, V2+Z2, V3+Z3, fz ) + O;
    }

    if( pos )
    {
        const Vec3f V0 = CatRom( POS[idx00]+X0, POS[idx10], POS[idx20]+X2, POS[idx30]+X3, fx );
        const Vec3f V1 = CatRom( POS[idx01]+X0, POS[idx11], POS[idx21]+X2, POS[idx31]+X3, fx );
        const Vec3f V2 = CatRom( POS[idx02]+X0, POS[idx12], POS[idx22]+X2, POS[idx32]+X3, fx );
        const Vec3f V3 = CatRom( POS[idx03]+X0, POS[idx13], POS[idx23]+X2, POS[idx33]+X3, fx );

        *pos = CatRom( V0+Z0, V1, V2+Z2, V3+Z3, fz ) + O;
    }

    if( nrm )
    {
        const Vec3f V0 = CatRom( NRM[idx00], NRM[idx10], NRM[idx20], NRM[idx30], fx );
        const Vec3f V1 = CatRom( NRM[idx01], NRM[idx11], NRM[idx21], NRM[idx31], fx );
        const Vec3f V2 = CatRom( NRM[idx02], NRM[idx12], NRM[idx22], NRM[idx32], fx );
        const Vec3f V3 = CatRom( NRM[idx03], NRM[idx13], NRM[idx23], NRM[idx33], fx );

        *nrm = CatRom( V0, V1, V2, V3, fz );
    }

    if( wav )
    {
        const float V0 = CatRom( WAV[idx00], WAV[idx10], WAV[idx20], WAV[idx30], fx );
        const float V1 = CatRom( WAV[idx01], WAV[idx11], WAV[idx21], WAV[idx31], fx );
        const float V2 = CatRom( WAV[idx02], WAV[idx12], WAV[idx22], WAV[idx32], fx );
        const float V3 = CatRom( WAV[idx03], WAV[idx13], WAV[idx23], WAV[idx33], fx );

        *wav = CatRom( V0, V1, V2, V3, fz );
    }

    if( crs )
    {
        const float V0 = CatRom( CRS[idx00], CRS[idx10], CRS[idx20], CRS[idx30], fx );
        const float V1 = CatRom( CRS[idx01], CRS[idx11], CRS[idx21], CRS[idx31], fx );
        const float V2 = CatRom( CRS[idx02], CRS[idx12], CRS[idx22], CRS[idx32], fx );
        const float V3 = CatRom( CRS[idx03], CRS[idx13], CRS[idx23], CRS[idx33], fx );

        *crs = CatRom( V0, V1, V2, V3, fz );
    }
}

void OceanTileVertexData::buildMesh( const int N, const float L )
{
    if( ( _N == N ) && ( _L == L ) )
    {
        return;
    }

    _N = N;
    _L = L;

    const float& h = _h = L / (float)N;

    eps = h*0.001f;

    const int numNodes = (N+1) * (N+1);

    GRD.resize( numNodes );
    POS.resize( numNodes );
    NRM.resize( numNodes );
    WAV.resize( numNodes );
    CRS.resize( numNodes );
    TRI.resize( 3*(2*N*N) );

    tmpPOS.resize( numNodes );
    tmpNRM.resize( numNodes );
    tmpWAV.resize( numNodes );
    tmpCRS.resize( numNodes );

    int index = 0;
    for( int j=0; j<N+1; ++j )
    {
        const float y = j*h;

        for( int i=0; i<N+1; ++i, ++index )
        {
            const float x = i*h;

            Vec3f& p = GRD[index];
            Vec3f& q = POS[index];

            p.x = q.x = x;
            p.y = q.y = 0.f;
            p.z = q.z = y;
        }
    }

    index = 0;
    for( int j=0; j<N; ++j )
    {
        for( int i=0; i<N; ++i, index+=6 )
        {
            // the first triangle
            TRI[index  ] = OceanTileVertexData::vertexIndex( i  , j   );
            TRI[index+1] = OceanTileVertexData::vertexIndex( i  , j+1 );
            TRI[index+2] = OceanTileVertexData::vertexIndex( i+1, j   );

            // the second triangle
            TRI[index+3] = OceanTileVertexData::vertexIndex( i+1, j   );
            TRI[index+4] = OceanTileVertexData::vertexIndex( i  , j+1 );
            TRI[index+5] = OceanTileVertexData::vertexIndex( i+1, j+1 );
        }
    }
}

BORA_NAMESPACE_END

