//---------------//
// StringUtils.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.11.22                               //
//-------------------------------------------------------//

#ifndef _BoraStringUtils_h_
#define _BoraStringUtils_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

inline bool IsUpper( const char c )
{
	if( c < 'A' ) { return false; }
	if( c > 'Z' ) { return false; }
	return true;
}

inline bool IsLower( const char c )
{
	if( c < 'a' ) { return false; }
	if( c > 'z' ) { return false; }
	return true;
}

inline bool IsDigit( const char c )
{
	if( c < '0' ) { return false; }
	if( c > '9' ) { return false; }
	return true;
}

inline char ToUpper( const char c )
{
	if( c < 'a' ) { return c; }
	if( c > 'z' ) { return c; }
	return (c-32);
}

inline char ToLower( const char c )
{
	if( c < 'A' ) { return c; }
	if( c > 'Z' ) { return c; }
	return (c+32);
}

int FirstIndexOf( const std::string& str, const char c );
int LastIndexOf( const std::string& str, const char c );

std::string SubString( const std::string& str, int startIndex, int endIndex );

bool IsAllDigit( const std::string& str );
bool IsAllAlpha( const std::string& str );
bool IsAllAlnum( const std::string& str );
bool IsAllLower( const std::string& str );
bool IsAllUpper( const std::string& str );

void Lowerize( std::string& str );
void Upperize( std::string& str );
void Capitalize( std::string& str );
void SwapCase( std::string& str );

void Replace( std::string& str, const char fromChar, const char toChar );
void Replace( std::string& str, const char* fromStr, const char* toStr );

void Reverse( std::string& str );

std::string Commify( int64_t number );

int Split( const std::string& str, const std::string& delimiter, std::vector<std::string>& tokens );

void RemoveSpace( std::string& str );

std::string MakePadding( const int number, const int padding );

size_t Count( const std::string& str, const char c );

bool ReadFromTextFile( std::string& str, const char* filePathName );

BORA_NAMESPACE_END

#endif

