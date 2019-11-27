//-----------------//
// StringArray.cpp //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.04.18                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

StringArray::StringArray()
{
    // nothing to do
}

StringArray::StringArray( const int initialLength )
{
    parent::resize( initialLength );
}

void StringArray::reset()
{
    parent::clear();
}

void StringArray::setLength( const size_t n )
{
    parent::resize( n );
}

size_t StringArray::length() const
{
    return parent::size();
}

std::string& StringArray::first()
{
    return parent::at(0);
}

const std::string& StringArray::first() const
{
    return parent::at(0);
}

std::string& StringArray::last()
{
    return parent::at(parent::size()-1);
}

const std::string& StringArray::last() const
{
    return parent::at(parent::size()-1);
}

std::string& StringArray::append( const std::string& str )
{
    parent::push_back( str );
    return parent::back();
}

void StringArray::append( const StringArray& a )
{
    parent::resize( parent::size() + a.size() );
    std::copy( a.begin(), a.end(), parent::end()-a.size() );
}

bool StringArray::operator==( const StringArray& a )
{
    const size_t n = parent::size();
    const size_t m = a.size();

    if( n != m ) { return false; }

    for( size_t i=0; i<n; ++i )
    {
        if( parent::at(i) != a[i] ) { return false; }
    }

    return true;
}

bool StringArray::operator!=( const StringArray& a )
{
    const size_t n = parent::size();
    const size_t m = a.size();

    if( n != m ) { return true; }

    for( size_t i=0; i<n; ++i )
    {
        if( parent::at(i) != a[i] ) { return true; }
    }

    return false;
}

void StringArray::erase( const size_t index )
{
    const size_t n = parent::size();

    if( index >= n ) { return; }

    size_t i = 0;
    std::vector<std::string>::iterator itr = parent::begin();
    for( ; itr!=parent::end(); ++itr, ++i )
    {
        if( i == index )
        {
            itr = parent::erase( itr );
        }
    }
}

void StringArray::exclude( const std::string& str )
{
    std::vector<std::string>::iterator itr = parent::begin();
    for( ; itr!=parent::end(); ++itr )
    {
        if( *itr == str )
        {
            itr = parent::erase( itr );
        }
    }
}

int StringArray::remove( const IndexArray& indicesToBeDeleted )
{
    const size_t n = parent::size();
    const size_t m = indicesToBeDeleted.size();

    size_t numToDelete = 0;
    Array<char> deleteMask( n );

    for( size_t i=0; i<m; ++i )
    {
        const size_t& idx = indicesToBeDeleted[i];

        if( idx >= n ) { continue; }

        if( deleteMask[idx] ) { continue; }

        deleteMask[idx] = true;

        ++numToDelete;
    }

    if( !deleteMask.size() )
    {
        return n;
    }

    if( n == numToDelete )
    {
        parent::clear();
        return 0;
    }

    const size_t finalSize = n - numToDelete;

    StringArray tmp( finalSize );

    for( size_t i=0, count=0; i<n; ++i )
    {
        if( deleteMask[i] ) { continue; }

        tmp[count++] = parent::at(i);
    }

    parent::swap( tmp );

    return finalSize;
}

void StringArray::shuffle( unsigned int seed )
{
    std::srand( seed );
    std::random_shuffle( parent::begin(), parent::end() );
}

void StringArray::reverse()
{
    std::reverse( parent::begin(), parent::end() );
}

void StringArray::exchange( StringArray& a )
{
    parent::swap( a );
}

size_t StringArray::setByTokenizing( const std::string& str, const std::string& delimiter )
{
    parent::clear();

    std::string::size_type lastPos = str.find_first_not_of( delimiter, 0 );
    std::string::size_type pos = str.find_first_of( delimiter, lastPos );

    while( str.npos != pos || str.npos != lastPos )
    {
        parent::push_back( str.substr( lastPos, pos - lastPos ) );
        lastPos = str.find_first_not_of( delimiter, pos );
        pos = str.find_first_of( delimiter, lastPos );
    }

    return parent::size();
}

std::string StringArray::combine( const std::string& separator ) const
{
    const size_t n = parent::size();

    std::string str( parent::at(0) );
    for( size_t i=1; i<n; ++i )
    {
        str += separator + parent::at(i);
    }

    return str;
}

void StringArray::write( std::ofstream& fout ) const
{
    const size_t n = parent::size();

    fout.write( (char*)&n, sizeof(size_t) );

    for( size_t i=0; i<n; ++i )
    {
        const std::string& str = parent::at(i);
        Write( str, fout );
    }
}

void StringArray::read( std::ifstream& fin )
{
    parent::clear();

    size_t n = 0;
    fin.read( (char*)&n, sizeof(size_t) );
    if( !n ) { return; }

    std::string str;

    for( size_t i=0; i<n; ++i )
    {
        Read( str, fin );
        parent::push_back( str );
    }
}


bool StringArray::save( const char* filePathName ) const
{
    std::ofstream fout( filePathName, std::ios::out|std::ios::binary|std::ios::trunc );

    if( fout.fail() || !fout.is_open() )
    {
        COUT << "Error@StringArray::save(): Failed to save file: " << filePathName << ENDL;
        return false;
    }

    StringArray::write( fout );

    fout.close();

    return true;
}

bool StringArray::load( const char* filePathName )
{
    std::ifstream fin( filePathName, std::ios::in|std::ios::binary );

    if( fin.fail() )
    {
        COUT << "Error@StringArray::load(): Failed to load file." << ENDL;
        parent::clear();
        return false;
    }

    StringArray::read( fin );

    fin.close();

    return true;
}

void StringArray::print( const bool horizontally, const bool withIndex, const size_t maxIndex ) const
{
    const size_t n = Min( parent::size(), maxIndex );

    if( horizontally )
    {
        for( size_t i=0; i<n; ++i )
        {
            if( withIndex ) { COUT << i << ": " << parent::at(i) << " "; }
            else { COUT << parent::at(i) << " "; }
        }
        COUT << ENDL;
    }
    else // vertically
    {
        for( size_t i=0; i<n; ++i )
        {
            if( withIndex ) { COUT << i << ": " << parent::at(i) << ENDL; }
            else { COUT << parent::at(i) << ENDL; }
        }
    }

    COUT << ENDL;
}

std::ostream& operator<<( std::ostream& os, const StringArray& object )
{
	os << "<StringArray>" << ENDL;
	os << " Size    : " << object.size()     << ENDL;
	os << " Capacity: " << object.capacity() << ENDL;
	os << ENDL;
	return os;
}

BORA_NAMESPACE_END

