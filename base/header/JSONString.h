//--------------//
// JSONString.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2018.03.09                               //
//-------------------------------------------------------//

#ifndef _JSONString_h_
#define _JSONString_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

class JSONString
{
    private:

        std::map<std::string, int>         intData;
        std::map<std::string, float>       floatData;
        std::map<std::string, Vec3f>       vectorData;
        std::map<std::string, std::string> stringData;

    public:

        JSONString();

        void clear();

        void append( const char* name, const int&         value );
        void append( const char* name, const float&       value );
        void append( const char* name, const double&      value );
        void append( const char* name, const Vec3f&       value );
        void append( const char* name, const std::string& value );

        bool get( const char* name, int&         value );
        bool get( const char* name, float&       value );
        bool get( const char* name, double&      value );
        bool get( const char* name, Vec3f&       value );
        bool get( const char* name, std::string& value );

        size_t numItems() const;

        std::string json() const;

        void get( std::string& str ) const;
        void set( const std::string& str );

        bool save( const char* filePathName ) const;
        bool load( const char* filePathName );
};

BORA_NAMESPACE_END

#endif

