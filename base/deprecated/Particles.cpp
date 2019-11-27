//---------------//
// Particles.cpp //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.10.27                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

Particles::Particles()
{
    // nothing to do
}

Particles::~Particles()
{
    Particles::reset();
}

void Particles::clear()
{
    std::vector<void*>::iterator itr = _arrayPtrs.begin();
    for( int i=0; itr!=_arrayPtrs.end(); ++itr, ++i )
    {
        const DataType& dataType = _dataTypes[i];

        if( dataType == DataType::kChar    ) { CharArray*   ptr = (CharArray*  )(*itr); delete ptr; continue; }
        if( dataType == DataType::kUInt64  ) { IndexArray*  ptr = (IndexArray* )(*itr); delete ptr; continue; }
        if( dataType == DataType::kFloat32 ) { FloatArray*  ptr = (FloatArray* )(*itr); delete ptr; continue; }
        if( dataType == DataType::kVec2f   ) { Vec2fArray*  ptr = (Vec2fArray* )(*itr); delete ptr; continue; }
        if( dataType == DataType::kVec3f   ) { Vec3fArray*  ptr = (Vec3fArray* )(*itr); delete ptr; continue; }
        if( dataType == DataType::kQuatf   ) { QuatfArray*  ptr = (QuatfArray* )(*itr); delete ptr; continue; }
        if( dataType == DataType::kMat33f  ) { Mat33fArray* ptr = (Mat33fArray*)(*itr); delete ptr; continue; }
        if( dataType == DataType::kMat44f  ) { Mat44fArray* ptr = (Mat44fArray*)(*itr); delete ptr; continue; }

        COUT << "Error@Particles::reset(): Invalid data type." << ENDL;
    }

    _aabb.reset();

    _attrNames.clear();
    _dataTypes.clear();
    _arrayPtrs.clear();
}

void Particles::reset()
{
    Particles::clear();

    _groupId    = -1;
    _groupColor = Vec3f();
    _groupName  = "";
}

bool Particles::isSupportedDataType( const DataType dataType ) const
{
    if( dataType == DataType::kChar    ) { return true; }
    if( dataType == DataType::kUInt64  ) { return true; }
    if( dataType == DataType::kFloat32 ) { return true; }
    if( dataType == DataType::kVec2f   ) { return true; }
    if( dataType == DataType::kVec3f   ) { return true; }
    if( dataType == DataType::kQuatf   ) { return true; }
    if( dataType == DataType::kMat33f  ) { return true; }
    if( dataType == DataType::kMat44f  ) { return true; }

    return false;
}

bool Particles::addAttribute( const char* attrName, const DataType dataType )
{
    if( Particles::isSupportedDataType( dataType ) == false )
    {
        COUT << "Error@Particles::addAttribute(): Invalid data type." << ENDL;
        return false;
    }

    const int attrIdx = Particles::attributeIndex( attrName );

    if( attrIdx >= 0 )
    {
        if( _dataTypes[attrIdx] == dataType )
        {
            return true;
        }
        else
        {
            COUT << "Error@Particles::addAttribute(): Already exists." << ENDL;
            return false;
        }
    }

    _attrNames.push_back( attrName );
    _dataTypes.push_back( dataType );

    if( dataType == DataType::kChar    ) { CharArray*   ptr = new CharArray();   _arrayPtrs.push_back(ptr); }
    if( dataType == DataType::kUInt64  ) { IndexArray*  ptr = new IndexArray();  _arrayPtrs.push_back(ptr); }
    if( dataType == DataType::kFloat32 ) { FloatArray*  ptr = new FloatArray();  _arrayPtrs.push_back(ptr); }
    if( dataType == DataType::kVec2f   ) { Vec2fArray*  ptr = new Vec2fArray();  _arrayPtrs.push_back(ptr); }
    if( dataType == DataType::kVec3f   ) { Vec3fArray*  ptr = new Vec3fArray();  _arrayPtrs.push_back(ptr); }
    if( dataType == DataType::kQuatf   ) { QuatfArray*  ptr = new QuatfArray();  _arrayPtrs.push_back(ptr); }
    if( dataType == DataType::kMat33f  ) { Mat33fArray* ptr = new Mat33fArray(); _arrayPtrs.push_back(ptr); }
    if( dataType == DataType::kMat44f  ) { Mat44fArray* ptr = new Mat44fArray(); _arrayPtrs.push_back(ptr); }

    return true;
}

size_t Particles::count() const
{
    size_t nParticles = 0;
    {
        const DataType& dataType = _dataTypes[0];

        if( dataType == DataType::kChar    ) { CharArray*   ptr = (CharArray*  )_arrayPtrs[0]; nParticles = ptr->size(); }
        if( dataType == DataType::kUInt64  ) { IndexArray*  ptr = (IndexArray* )_arrayPtrs[0]; nParticles = ptr->size(); }
        if( dataType == DataType::kFloat32 ) { FloatArray*  ptr = (FloatArray* )_arrayPtrs[0]; nParticles = ptr->size(); }
        if( dataType == DataType::kVec2f   ) { Vec2fArray*  ptr = (Vec2fArray* )_arrayPtrs[0]; nParticles = ptr->size(); }
        if( dataType == DataType::kVec3f   ) { Vec3fArray*  ptr = (Vec3fArray* )_arrayPtrs[0]; nParticles = ptr->size(); }
        if( dataType == DataType::kQuatf   ) { QuatfArray*  ptr = (QuatfArray* )_arrayPtrs[0]; nParticles = ptr->size(); }
        if( dataType == DataType::kMat33f  ) { Mat33fArray* ptr = (Mat33fArray*)_arrayPtrs[0]; nParticles = ptr->size(); }
        if( dataType == DataType::kMat44f  ) { Mat44fArray* ptr = (Mat44fArray*)_arrayPtrs[0]; nParticles = ptr->size(); }
    }

    bool different = false;

    std::vector<void*>::const_iterator itr = _arrayPtrs.begin();
    for( int i=0; itr!=_arrayPtrs.end(); ++itr, ++i )
    {
        const DataType& dataType = _dataTypes[i];

        if( dataType == DataType::kChar    ) { const CharArray*   ptr = (CharArray*  )(*itr); if(nParticles!=ptr->size()){different=true;} }
        if( dataType == DataType::kUInt64  ) { const IndexArray*  ptr = (IndexArray* )(*itr); if(nParticles!=ptr->size()){different=true;} }
        if( dataType == DataType::kFloat32 ) { const FloatArray*  ptr = (FloatArray* )(*itr); if(nParticles!=ptr->size()){different=true;} }
        if( dataType == DataType::kVec2f   ) { const Vec2fArray*  ptr = (Vec2fArray* )(*itr); if(nParticles!=ptr->size()){different=true;} }
        if( dataType == DataType::kVec3f   ) { const Vec3fArray*  ptr = (Vec3fArray* )(*itr); if(nParticles!=ptr->size()){different=true;} }
        if( dataType == DataType::kQuatf   ) { const QuatfArray*  ptr = (QuatfArray* )(*itr); if(nParticles!=ptr->size()){different=true;} }
        if( dataType == DataType::kMat33f  ) { const Mat33fArray* ptr = (Mat33fArray*)(*itr); if(nParticles!=ptr->size()){different=true;} }
        if( dataType == DataType::kMat44f  ) { const Mat44fArray* ptr = (Mat44fArray*)(*itr); if(nParticles!=ptr->size()){different=true;} }
    }

    if( different )
    {
        COUT << "Error@Particles::count(): Different array lengths." << ENDL;
    }

    return nParticles;
}

int Particles::numAttributes() const
{
    return (int)_attrNames.size();
}

int Particles::attributeIndex( const char* attrName ) const
{
    const int nAttrs = Particles::numAttributes();

    for( int i=0; i<nAttrs; ++i )
    {
        if( _attrNames[i] == attrName )
        {
            return i;
        }
    }

    return -1;
}

std::string Particles::attributeName( const int attrIndex ) const
{
    if( ( attrIndex < 0 ) || ( attrIndex >= Particles::numAttributes() ) )
    {
        return "";
    }

    return _attrNames[attrIndex];
}

DataType Particles::dataType( const int attrIndex ) const
{
    if( ( attrIndex < 0 ) || ( attrIndex >= Particles::numAttributes() ) )
    {
        return DataType::kNone;
    }

    return _dataTypes[attrIndex];
}

bool Particles::hasPosition() const
{
    const int i = Particles::attributeIndex( "position" );

    if( i < 0 ) { return false; }
    if( _dataTypes[i] != DataType::kVec3f ) { return false; }

    return true;
}

Vec3fArray& Particles::position()
{
    const int i = Particles::attributeIndex( "position" );
    return (*(Vec3fArray*)_arrayPtrs[i]);
}

const Vec3fArray& Particles::position() const
{
    const int i = Particles::attributeIndex( "position" );
    return (*(Vec3fArray*)_arrayPtrs[i]);
}

bool Particles::hasVelocity() const
{
    const int i = Particles::attributeIndex( "velocity" );

    if( i < 0 ) { return false; }
    if( _dataTypes[i] != DataType::kVec3f ) { return false; }

    return true;
}

Vec3fArray& Particles::velocity()
{
    const int i = Particles::attributeIndex( "velocity" );
    return (*(Vec3fArray*)_arrayPtrs[i]);
}

const Vec3fArray& Particles::velocity() const
{
    const int i = Particles::attributeIndex( "velocity" );
    return (*(Vec3fArray*)_arrayPtrs[i]);
}

bool Particles::updateBoundingBox()
{
    if( Particles::hasPosition() == false )
    {
        COUT << "Error@Particles::updateBoundingBox(): Failed to get the attribute." << ENDL;
        return false;
    }

    const Vec3fArray& positions = Particles::position();

    _aabb = BoundingBox( positions );

    return true;
}

const AABB3f Particles::boundingBox() const
{
    return _aabb;
}

Vec3f Particles::maxVelocity() const
{
    if( Particles::hasVelocity() == false )
    {
        COUT << "Error@Particles::updateBoundingBox(): Failed to get the attribute." << ENDL;
        return false;
    }

    const Vec3fArray& velocity = Particles::velocity();

    return MaxMagVector( velocity );
}

void Particles::remove( const IndexArray& indicesToBeDeleted )
{
    if( Particles::numAttributes() == 0 )
    {
        return;
    }

    CharArray deleteMask;
    size_t numToDelete = 0;
    {
        const DataType& dataType = _dataTypes[0];

        if( dataType == DataType::kChar    ) { CharArray*   ptr = (CharArray*  )_arrayPtrs[0]; numToDelete = ptr->buildDeleteMask( indicesToBeDeleted, deleteMask ); }
        if( dataType == DataType::kUInt64  ) { IndexArray*  ptr = (IndexArray* )_arrayPtrs[0]; numToDelete = ptr->buildDeleteMask( indicesToBeDeleted, deleteMask ); }
        if( dataType == DataType::kFloat32 ) { FloatArray*  ptr = (FloatArray* )_arrayPtrs[0]; numToDelete = ptr->buildDeleteMask( indicesToBeDeleted, deleteMask ); }
        if( dataType == DataType::kVec2f   ) { Vec2fArray*  ptr = (Vec2fArray* )_arrayPtrs[0]; numToDelete = ptr->buildDeleteMask( indicesToBeDeleted, deleteMask ); }
        if( dataType == DataType::kVec3f   ) { Vec3fArray*  ptr = (Vec3fArray* )_arrayPtrs[0]; numToDelete = ptr->buildDeleteMask( indicesToBeDeleted, deleteMask ); }
        if( dataType == DataType::kQuatf   ) { QuatfArray*  ptr = (QuatfArray* )_arrayPtrs[0]; numToDelete = ptr->buildDeleteMask( indicesToBeDeleted, deleteMask ); }
        if( dataType == DataType::kMat33f  ) { Mat33fArray* ptr = (Mat33fArray*)_arrayPtrs[0]; numToDelete = ptr->buildDeleteMask( indicesToBeDeleted, deleteMask ); }
        if( dataType == DataType::kMat44f  ) { Mat44fArray* ptr = (Mat44fArray*)_arrayPtrs[0]; numToDelete = ptr->buildDeleteMask( indicesToBeDeleted, deleteMask ); }
    }

    std::vector<void*>::iterator itr = _arrayPtrs.begin();
    for( int i=0; itr!=_arrayPtrs.end(); ++itr, ++i )
    {
        const DataType& dataType = _dataTypes[i];

        if( dataType == DataType::kChar    ) { CharArray*   ptr = (CharArray*  )(*itr); ptr->remove( deleteMask, numToDelete ); continue; }
        if( dataType == DataType::kUInt64  ) { IndexArray*  ptr = (IndexArray* )(*itr); ptr->remove( deleteMask, numToDelete ); continue; }
        if( dataType == DataType::kFloat32 ) { FloatArray*  ptr = (FloatArray* )(*itr); ptr->remove( deleteMask, numToDelete ); continue; }
        if( dataType == DataType::kVec2f   ) { Vec2fArray*  ptr = (Vec2fArray* )(*itr); ptr->remove( deleteMask, numToDelete ); continue; }
        if( dataType == DataType::kVec3f   ) { Vec3fArray*  ptr = (Vec3fArray* )(*itr); ptr->remove( deleteMask, numToDelete ); continue; }
        if( dataType == DataType::kQuatf   ) { QuatfArray*  ptr = (QuatfArray* )(*itr); ptr->remove( deleteMask, numToDelete ); continue; }
        if( dataType == DataType::kMat33f  ) { Mat33fArray* ptr = (Mat33fArray*)(*itr); ptr->remove( deleteMask, numToDelete ); continue; }
        if( dataType == DataType::kMat44f  ) { Mat44fArray* ptr = (Mat44fArray*)(*itr); ptr->remove( deleteMask, numToDelete ); continue; }
    }
}

bool Particles::save( const char* filePathName ) const
{
    std::ofstream fout( filePathName, std::ios::out|std::ios::binary|std::ios::trunc );

    if( fout.fail() || !fout.is_open() )
    {
        COUT << "Error@Particles::save(): Failed to save file: " << filePathName << ENDL;
        return false;
    }

    _aabb.write( fout );

    _attrNames.write( fout );

    std::vector<void*>::const_iterator itr = _arrayPtrs.begin();
    for( int i=0; itr!=_arrayPtrs.end(); ++itr, ++i )
    {
        const DataType& dataType = _dataTypes[i];

        fout.write( (char*)&dataType, sizeof(DataType) );

        if( dataType == DataType::kChar    ) { const CharArray*   ptr = (CharArray*  )(*itr); ptr->write( fout ); }
        if( dataType == DataType::kUInt64  ) { const IndexArray*  ptr = (IndexArray* )(*itr); ptr->write( fout ); }
        if( dataType == DataType::kFloat32 ) { const FloatArray*  ptr = (FloatArray* )(*itr); ptr->write( fout ); }
        if( dataType == DataType::kVec2f   ) { const Vec2fArray*  ptr = (Vec2fArray* )(*itr); ptr->write( fout ); }
        if( dataType == DataType::kVec3f   ) { const Vec3fArray*  ptr = (Vec3fArray* )(*itr); ptr->write( fout ); }
        if( dataType == DataType::kQuatf   ) { const QuatfArray*  ptr = (QuatfArray* )(*itr); ptr->write( fout ); }
        if( dataType == DataType::kMat33f  ) { const Mat33fArray* ptr = (Mat33fArray*)(*itr); ptr->write( fout ); }
        if( dataType == DataType::kMat44f  ) { const Mat44fArray* ptr = (Mat44fArray*)(*itr); ptr->write( fout ); }
    }

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

    _aabb.read( fin );

    _attrNames.read( fin );

    const int nAttrs = (int)_attrNames.length();

    for( int i=0; i<nAttrs; ++i )
    {
        DataType dataType = DataType::kNone;

        fin.read( (char*)&dataType, sizeof(DataType) );

        if( dataType == DataType::kChar    ) { CharArray*   ptr = new CharArray();   ptr->read(fin); _arrayPtrs.push_back(ptr); }
        if( dataType == DataType::kUInt64  ) { IndexArray*  ptr = new IndexArray();  ptr->read(fin); _arrayPtrs.push_back(ptr); }
        if( dataType == DataType::kFloat32 ) { FloatArray*  ptr = new FloatArray();  ptr->read(fin); _arrayPtrs.push_back(ptr); }
        if( dataType == DataType::kVec2f   ) { Vec2fArray*  ptr = new Vec2fArray();  ptr->read(fin); _arrayPtrs.push_back(ptr); }
        if( dataType == DataType::kVec3f   ) { Vec3fArray*  ptr = new Vec3fArray();  ptr->read(fin); _arrayPtrs.push_back(ptr); }
        if( dataType == DataType::kQuatf   ) { QuatfArray*  ptr = new QuatfArray();  ptr->read(fin); _arrayPtrs.push_back(ptr); }
        if( dataType == DataType::kMat33f  ) { Mat33fArray* ptr = new Mat33fArray(); ptr->read(fin); _arrayPtrs.push_back(ptr); }
        if( dataType == DataType::kMat44f  ) { Mat44fArray* ptr = new Mat44fArray(); ptr->read(fin); _arrayPtrs.push_back(ptr); }
    }

    fin.close();

    return true;
}

std::ostream& operator<<( std::ostream& os, const Particles& ptc )
{
    const int nAttrs = ptc.numAttributes();

	os << "<Particles>" << ENDL;

    os << " Count: " << ptc.count() << ENDL;

    for( int i=0; i<nAttrs; ++i )
    {
        const std::string attrName = ptc.attributeName(i);
        const DataType dataType = ptc.dataType(i);

        os << i << ": " << attrName << " (" << DataName( dataType ) << ")" << ENDL;
    }

	os << ENDL;
	return os;
}

BORA_NAMESPACE_END

