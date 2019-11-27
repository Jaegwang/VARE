//---------------//
// StringArray.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.04.18                               //
//-------------------------------------------------------//

#ifndef _BoraStringArray_h_
#define _BoraStringArray_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

class StringArray : public std::vector<std::string>
{
    private:

        typedef std::vector<std::string> parent;

    public:

        StringArray();

        StringArray( const int initialLength );

        void reset();

        void setLength( const size_t n );

        size_t length() const;

        std::string& first();
        const std::string& first() const;

        std::string& last();
        const std::string& last() const;

        std::string& append( const std::string& str );
        void append( const StringArray& a );

        bool operator==( const StringArray& a );
        bool operator!=( const StringArray& a );

        void erase( const size_t index );
        void exclude( const std::string& str );
        int remove( const IndexArray& indicesToBeDeleted );

        void shuffle( unsigned int seed=0 );

        void reverse();

        void exchange( StringArray& a );

        size_t setByTokenizing( const std::string& str, const std::string& delimiter );
        std::string combine( const std::string& separator="" ) const;

        void write( std::ofstream& fout ) const;
        void read( std::ifstream& fin );

        bool save( const char* filePathName ) const;
        bool load( const char* filePathName );

        void print( const bool horizontally=false, const bool withIndex=true, const size_t maxIndex=ULONG_MAX ) const;
};

std::ostream& operator<<( std::ostream& os, const StringArray& object );

BORA_NAMESPACE_END

#endif

