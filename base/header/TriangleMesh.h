//----------------//
// TriangleMesh.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.03.09                               //
//-------------------------------------------------------//

#ifndef _BoraTriangleMesh_h_
#define _BoraTriangleMesh_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

class TriangleMesh : public Particles
{
    public:

        IndexArray indices; // triangle indices in world space

    public:

        TriangleMesh( const MemorySpace memorySpace=MemorySpace::kHost );

        TriangleMesh& operator=( const TriangleMesh& mesh );

        // Caustion) reset() and clear() are different.
        // reset() = clear() + alpha
        void reset();
        void clear();

        BORA_FUNC_QUAL
        size_t numVertices() const;

        BORA_FUNC_QUAL
        size_t numUVWs() const;

        BORA_FUNC_QUAL
        size_t numTriangles() const;

        BORA_FUNC_QUAL
        AABB3f boundingBox() const;

        void drawVertices() const;
        void drawWireframe() const;
        void drawUVW() const;

        bool save( const char* filePathName ) const;

        bool load( const char* filePathName );

//		void exportAlembic( std::string& filePath );
//		bool importAlembic( std::string& filePath );

};

std::ostream& operator<<( std::ostream& os, const TriangleMesh& object );

BORA_NAMESPACE_END

#endif

