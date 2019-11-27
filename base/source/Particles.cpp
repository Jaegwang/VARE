//---------------//
// Particles.cpp //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
//         Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.12.07                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

Particles::Particles( MemorySpace memorySpace )
: typ(type), stt(state), uid(uniqueId), idx(index), rad(radius), mss(mass), den(density), dst(distance)
, lfs(lifespan), pos(position), prv(previous), vel(velocity), acc(acceleration), frc(force), amn(angularMomentum), vrt(vorticity)
, nrm(normal), clr(color), uvw(textureCoordinates), ort(orientation), dgr(deformationGradient)
, dcr(drawingColor)
{
    Particles::initialize( memorySpace );
}

void Particles::initialize( MemorySpace memorySpace )
{
    groupId    = 0;
    groupColor = Vec3f( 1.f, 1.f, 1.f );
    timeScale  = 1.f;

    typ.initialize( 0, memorySpace );
    stt.initialize( 0, memorySpace );
    uid.initialize( 0, memorySpace );
    idx.initialize( 0, memorySpace );
    rad.initialize( 0, memorySpace );
    mss.initialize( 0, memorySpace );
    den.initialize( 0, memorySpace );
    dst.initialize( 0, memorySpace );
    age.initialize( 0, memorySpace );
    lfs.initialize( 0, memorySpace );
    pos.initialize( 0, memorySpace );
    prv.initialize( 0, memorySpace );
    vel.initialize( 0, memorySpace );
    acc.initialize( 0, memorySpace );
    frc.initialize( 0, memorySpace );
    amn.initialize( 0, memorySpace );
    vrt.initialize( 0, memorySpace );
    nrm.initialize( 0, memorySpace );
    clr.initialize( 0, memorySpace );
    uvw.initialize( 0, memorySpace );
    ort.initialize( 0, memorySpace );
    dgr.initialize( 0, memorySpace );
    dcr.initialize( 0, memorySpace );
}

Particles& Particles::operator=( const Particles& ptc )
{
    aabb       = ptc.aabb;
    groupId    = ptc.groupId;
    groupColor = ptc.groupColor;
    timeScale  = ptc.timeScale;

    typ = ptc.typ;
    stt = ptc.stt;
    uid = ptc.uid;
    idx = ptc.idx;
    rad = ptc.rad;
    mss = ptc.mss;
    den = ptc.den;
    dst = ptc.dst;
    age = ptc.age;
    lfs = ptc.lfs;
    pos = ptc.pos;
    prv = ptc.prv;
    vel = ptc.vel;
    acc = ptc.acc;
    frc = ptc.frc;
    amn = ptc.amn;
    vrt = ptc.vrt;
    nrm = ptc.nrm;
    clr = ptc.clr;
    uvw = ptc.uvw;
    ort = ptc.ort;
    dgr = ptc.dgr;
    dcr = ptc.dcr;

    return (*this);
}

void Particles::reset()
{
    aabb.reset();

    groupId    = 0;
    groupColor = Vec3f( 1.f, 1.f, 1.f );
    timeScale  = 1.f;

    Particles::clear();
}

void Particles::clear()
{
    typ.clear();
    stt.clear();
    uid.clear();
    idx.clear();
    rad.clear();
    mss.clear();
    den.clear();
    dst.clear();
    age.clear();
    lfs.clear();
    pos.clear();
    prv.clear();
    vel.clear();
    acc.clear();
    frc.clear();
    amn.clear();
    vrt.clear();
    nrm.clear();
    clr.clear();
    uvw.clear();
    ort.clear();
    dgr.clear();
    dcr.clear();
}

void Particles::updateBoundingBox()
{
    aabb = BoundingBox( pos );
}

void Particles::remove( const Array<char>& deleteMask )
{
}

/*
void Particles::remove( const IndexArray& indicesToBeDeleted )
{
    CharArray deleteMask;
    const size_t numToDelete = pos.buildDeleteMask( indicesToBeDeleted, deleteMask );

    typ.remove( deleteMask, numToDelete );
    stt.remove( deleteMask, numToDelete );
    uid.remove( deleteMask, numToDelete );
    idx.remove( deleteMask, numToDelete );
    rad.remove( deleteMask, numToDelete );
    mss.remove( deleteMask, numToDelete );
    den.remove( deleteMask, numToDelete );
    dst.remove( deleteMask, numToDelete );
    age.remove( deleteMask, numToDelete );
    lfs.remove( deleteMask, numToDelete );
    pos.remove( deleteMask, numToDelete );
    prv.remove( deleteMask, numToDelete );
    vel.remove( deleteMask, numToDelete );
    acc.remove( deleteMask, numToDelete );
    frc.remove( deleteMask, numToDelete );
    amn.remove( deleteMask, numToDelete );
    vrt.remove( deleteMask, numToDelete );
    nrm.remove( deleteMask, numToDelete );
    clr.remove( deleteMask, numToDelete );
    uvw.remove( deleteMask, numToDelete );
    ort.remove( deleteMask, numToDelete );
    dgr.remove( deleteMask, numToDelete );
    dcr.remove( deleteMask, numToDelete );
}
*/

void Particles::draw() const
{
    if( pos.size() == 0 ) { return; }

    const size_t n = pos.size();

    const bool withColor = ( dcr.size() == n ) ? true : false;
 
    glEnableClientState( GL_VERTEX_ARRAY );

    if( withColor )
    {
        glEnableClientState( GL_COLOR_ARRAY  );
        glColorPointer ( 4, GL_FLOAT, 0, &dcr[0].r );
    }

    glVertexPointer( 3, GL_FLOAT, 0, &pos.at(0) );
    glDrawArrays( GL_POINTS, 0, pos.size() );    
    glDisableClientState( GL_VERTEX_ARRAY );

    if( withColor )
    {
        glDisableClientState( GL_COLOR_ARRAY  );
    }
}

void Particles::write( std::ofstream& fout ) const
{
    aabb.write( fout );

    fout.write( (char*)&groupId, sizeof(int) );
    groupColor.write( fout );
    fout.write( (char*)&timeScale, sizeof(float) );

    typ.write( fout );
    stt.write( fout );
    uid.write( fout );
    idx.write( fout );
    rad.write( fout );
    mss.write( fout );
    den.write( fout );
    dst.write( fout );
    age.write( fout );
    lfs.write( fout );
    pos.write( fout );
    prv.write( fout );
    vel.write( fout );
    acc.write( fout );
    frc.write( fout );
    amn.write( fout );
    vrt.write( fout );
    nrm.write( fout );
    clr.write( fout );
    uvw.write( fout );
    ort.write( fout );
    dgr.write( fout );
}

void Particles::read( std::ifstream& fin )
{
    aabb.read( fin );

    fin.read( (char*)&groupId, sizeof(int) );
    groupColor.read( fin );
    fin.read( (char*)&timeScale, sizeof(float) );

    typ.read( fin );
    stt.read( fin );
    uid.read( fin );
    idx.read( fin );
    rad.read( fin );
    mss.read( fin );
    den.read( fin );
    dst.read( fin );
    age.read( fin );
    lfs.read( fin );
    pos.read( fin );
    prv.read( fin );
    vel.read( fin );
    acc.read( fin );
    frc.read( fin );
    amn.read( fin );
    vrt.read( fin );
    nrm.read( fin );
    clr.read( fin );
    uvw.read( fin );
    ort.read( fin );
    dgr.read( fin );
}

bool Particles::save( const char* filePathName ) const
{
    std::ofstream fout( filePathName, std::ios::out|std::ios::binary|std::ios::trunc );

    if( fout.fail() || !fout.is_open() )
    {
        COUT << "Error@Particles::save(): Failed to save file: " << filePathName << ENDL;
        return false;
    }

    Particles::write( fout );

    fout.close();

    return true;
}

bool Particles::load( const char* filePathName )
{
    Particles::reset();

    std::ifstream fin( filePathName, std::ios::in|std::ios::binary );

    if( fin.fail() )
    {
        COUT << "Error@Particles::load(): Failed to load file." << ENDL;
        return false;
    }

    Particles::read( fin );

    fin.close();

    return true;
}

std::ostream& operator<<( std::ostream& os, const Particles& object )
{
    os << "<Particles>" << ENDL;
    os << " Type               : " << ( object.typ.size() ? "O" : "X" ) << ENDL;
    os << " State              : " << ( object.stt.size() ? "O" : "X" ) << ENDL;
    os << " UniqueId           : " << ( object.uid.size() ? "O" : "X" ) << ENDL;
    os << " Index              : " << ( object.idx.size() ? "O" : "X" ) << ENDL;
    os << " Radius             : " << ( object.rad.size() ? "O" : "X" ) << ENDL;
    os << " Mass               : " << ( object.mss.size() ? "O" : "X" ) << ENDL;
    os << " Density            : " << ( object.den.size() ? "O" : "X" ) << ENDL;
    os << " SignedDistance     : " << ( object.dst.size() ? "O" : "X" ) << ENDL;
    os << " Age                : " << ( object.age.size() ? "O" : "X" ) << ENDL;
    os << " Lifespan           : " << ( object.lfs.size() ? "O" : "X" ) << ENDL;
    os << " Position           : " << ( object.pos.size() ? "O" : "X" ) << ENDL;
    os << " Previous           : " << ( object.prv.size() ? "O" : "X" ) << ENDL;
    os << " Velocity           : " << ( object.vel.size() ? "O" : "X" ) << ENDL;
    os << " Acceleration       : " << ( object.acc.size() ? "O" : "X" ) << ENDL;
    os << " Force              : " << ( object.frc.size() ? "O" : "X" ) << ENDL;
    os << " Angular Momentum   : " << ( object.amn.size() ? "O" : "X" ) << ENDL;
    os << " Vorticity          : " << ( object.vrt.size() ? "O" : "X" ) << ENDL;
    os << " Normal             : " << ( object.nrm.size() ? "O" : "X" ) << ENDL;
    os << " Color              : " << ( object.clr.size() ? "O" : "X" ) << ENDL;
    os << " TextureCoordinates : " << ( object.uvw.size() ? "O" : "X" ) << ENDL;
    os << " Orientation        : " << ( object.ort.size() ? "O" : "X" ) << ENDL;
    os << " DeformationGradient: " << ( object.dgr.size() ? "O" : "X" ) << ENDL;
    os << ENDL;
    return os;
}

BORA_NAMESPACE_END

