//-----------------//
// StringUtils.cpp //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.11.22                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

int FirstIndexOf( const std::string& str, const char c )
{
    return str.find_first_of( c );
}

int LastIndexOf( const std::string& str, const char c )
{
    return str.find_last_of( c );
}

std::string SubString( const std::string& str, int startIndex, int endIndex )
{
    const int n = endIndex - startIndex + 1;
    if( n <= 0 ) { return ""; }

    std::string substr;
    substr.resize( 3 );
    memcpy( (void*)&substr[0], (const void*)&str[startIndex], n*sizeof(char) );

    return substr;
}

bool IsAllDigit( const std::string& str )
{
    const int n = (int)str.length();

    for( int i=0; i<n; ++i )
    {
        const char& c = str[i];

        if( !IsDigit( c ) ) { return false; }
    }

    return true;
}

bool IsAllAlpha( const std::string& str )
{
    const int n = (int)str.length();

    for( int i=0; i<n; ++i )
    {
        const char& c = str[i];

        if( !IsLower(c) ) { return false; }
        if( !IsUpper(c) ) { return false; }
    }

    return true;
}

bool IsAllAlnum( const std::string& str )
{
    const int n = (int)str.length();

    for( int i=0; i<n; ++i )
    {
        const char& c = str[i];

        if( !IsDigit(c) ) { return false; }
        if( !IsLower(c) ) { return false; }
        if( !IsUpper(c) ) { return false; }
    }

    return true;
}

bool IsAllLower( const std::string& str )
{
    const int n = (int)str.length();

    for( int i=0; i<n; ++i )
    {
        const char& c = str[i];

        if( IsUpper( c ) ) { return false; }
    }

    return true;
}

bool IsAllUpper( const std::string& str )
{
    const int n = (int)str.length();

    for( int i=0; i<n; ++i )
    {
        const char& c = str[i];

        if( IsLower( c ) ) { return false; }
    }

    return true;
}

void Lowerize( std::string& str )
{
    const int n = (int)str.length();

    for( int i=0; i<n; ++i )
    {
        char& c = str[i];

        if( IsUpper( c ) ) { c = ToLower( c ); }
    }
}

void Upperize( std::string& str )
{
    const int n = (int)str.length();

    for( int i=0; i<n; ++i )
    {
        char& c = str[i];

        if( IsLower( c ) ) { c = ToUpper( c ); }
    }
}

void Capitalize( std::string& str )
{
    const int n = (int)str.length();

    if( n > 0 )
    {
        char& c = str[0];

        if( IsLower( c ) ) { c = ToUpper( c ); }
    }

    for( int i=1; i<n; ++i )
    {
        char& c = str[i];

        if( IsUpper( c ) ) { c = ToLower( c ); }
    }
}

void SwapCase( std::string& str )
{
    const int n = (int)str.length();

    for( int i=0; i<n; ++i )
    {
        char& c = str[i];

        if( IsLower( c ) ) { c = ToUpper( c ); }
        else if( IsUpper( c ) ) { c = ToLower( c ); }
    }
}

void Replace( std::string& str, const char fromChar, const char toChar )
{
    const int n = (int)str.length();

    for( int i=0; i<n; ++i )
    {
        if( str[i] == fromChar ) { str[i] = toChar; }
    }
}

void Replace( std::string& str, const char* fromStr, const char* toStr )
{
    const int fromLen = (int)strlen( fromStr );
    const int toLen   = (int)strlen( toStr );

    if( !fromLen || !toLen )
    {
        return;
    }

    int index = 0;

    while( (index=str.find(fromStr,index)) != std::string::npos )
    {
        str.replace( index, fromLen, toStr );
        index += toLen;
    }
}

void Reverse( std::string& str )
{
    const int n = (int)str.length();

    for( int i=0; i<(n/2); ++i )
    {
       Swap( str[i], str[n-1-i] );
    }
}

std::string Commify( int64_t number )
{
    bool minus = false;

    if( number < 0 )
    {
        minus = true;
        number *= -1;
    }

    std::stringstream ss;
    ss << number;
    std::string str = ss.str();

    int64_t len = (int64_t)str.size();
    if( len == 0 ) { return str; }

    int64_t numCommas = (int64_t)((len-1)/3);
    int64_t totalLen = len + numCommas;

    std::string result( totalLen, '.' );

    int64_t shift = -len;

    int64_t i=0, j=0;
    while( result[i] )
    {
        result[i++] = str[j++];
        if( ++shift && (shift%3) == 0 ) { result[i++] = ','; }
    }

    if( minus )
    {
        result.insert( 0, "-" );
    }

    return result;
}

int Split( const std::string& str, const std::string& delimiter, std::vector<std::string>& tokens )
{
    tokens.clear();

    std::string::size_type lastPos = str.find_first_not_of( delimiter, 0 );
    std::string::size_type pos = str.find_first_of( delimiter, lastPos );

    while( str.npos != pos || str.npos != lastPos )
    {
        tokens.push_back( str.substr( lastPos, pos - lastPos ) );
        lastPos = str.find_first_not_of( delimiter, pos );
        pos = str.find_first_of( delimiter, lastPos );
    }

    return (int)tokens.size();
}

void RemoveSpace( std::string& str )
{
    std::vector<std::string> tokens;
    Split( str, " ", tokens );

    str.clear();

    const int n = (int)tokens.size();

    for( int i=0; i<n; ++i )
    {
        str += tokens[i];
    }
}

std::string MakePadding( const int number, const int padding )
{
    std::stringstream oss;
    oss << std::setfill('0') << std::setw(padding) << number;
    return oss.str();
}

size_t Count( const std::string& str, const char c )
{
    return std::count( str.begin(), str.end(), c );
}

bool ReadFromTextFile( std::string& str, const char* filePathName )
{
    str.clear();

    char* text = (char*)NULL;

    // Open the file and move the current position to the end position of the file.
    std::ifstream file( filePathName, std::ios::in | std::ios::ate );

    if( file.fail() ) { return false; }

    const int textLength = file.tellg(); // the current position
    text = new char[textLength]; // memory allocation for storing text
    file.seekg( 0, std::ios::beg ); // move to the start position of the file
    file.read( text, textLength ); // read data
    file.close(); // close the file

    if( text ) { text[textLength] = '\0'; } // add an end line char at the end

    str = text; // copy to this ZString intance
    if( text ) { delete[] text; } // delete memories

    return true;
}

BORA_NAMESPACE_END

