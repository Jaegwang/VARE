//-------------//
// Particles.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.10.27                               //
//-------------------------------------------------------//

#ifndef _BoraParticles_h_
#define _BoraParticles_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

class Particles
{
    private:

        AABB3f _aabb;

        StringArray           _attrNames;
        std::vector<DataType> _dataTypes;
        std::vector<void*>    _arrayPtrs;

    public:

        int         _groupId = -1;
        Vec3f       _groupColor;
        std::string _groupName;

    public:

        Particles();

        virtual ~Particles();

        // Caution)
        // They are different.
        void clear();
        void reset();

        bool isSupportedDataType( const DataType dataType ) const;

        bool addAttribute( const char* attrName, const DataType dataType );

        size_t count() const;

        int numAttributes() const;

        int attributeIndex( const char* attrName ) const;
        std::string attributeName( const int attrIndex ) const;
        DataType dataType( const int attrIndex ) const;

        bool hasPosition() const;
        Vec3fArray& position();
        const Vec3fArray& position() const;

        bool hasVelocity() const;
        Vec3fArray& velocity();
        const Vec3fArray& velocity() const;

        bool updateBoundingBox();
        const AABB3f boundingBox() const;

        Vec3f maxVelocity() const;

        void remove( const IndexArray& indicesToBeDeleted );

        bool save( const char* filePathName ) const;
        bool load( const char* filePathName );
};

std::ostream& operator<<( std::ostream& os, const Particles& object );

BORA_NAMESPACE_END

#endif

