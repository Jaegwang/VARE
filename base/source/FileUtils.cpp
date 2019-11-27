//---------------//
// FileUtils.cpp //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.03.19                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

bool IsDirectory( const char* path )
{
    struct stat fstat;
    lstat( path, &fstat );
    if( S_ISDIR(fstat.st_mode) ) { return true; }
    return false;
}

bool IsSymbolicLink( const char* path )
{
    struct stat fstat;
    lstat( path, &fstat );
    if( S_ISLNK(fstat.st_mode) ) { return true; }
    return false;
}

bool DoesFileExist( const char* path )
{
    struct stat buffer;
    const int exist = stat( path, &buffer );
    return ( (exist==0) ? true : false );
}

size_t FileSize( const char* path )
{
    if( !DoesFileExist(path) ) { return 0; }

    struct stat buffer;

    stat( path, &buffer );

    return (size_t)buffer.st_size; // type -> size_t
}

bool CreateDirectory( const char* path, const mode_t permission )
{
    if( DoesFileExist( path ) ) { return true; }

    std::string tmp( path );

    StringArray tokens;
    const size_t N = tokens.setByTokenizing( tmp, "/" );

	if( N == 0 ) { return false; }

	tmp.clear();

    for( size_t i=0; i<N; ++i )
    {
        tmp += "/" + tokens[i];

        if( !DoesFileExist( tmp.c_str() ) )

        if( mkdir( tmp.c_str(), permission ) )
        {
            COUT << "Error@CreateDirectory(): Failed to create a directory." << ENDL;
            return false;
        }
    }

	return true;
}

bool DeleteDirectory( const char* path )
{
    if( IsDirectory( path ) == false )
    {
        COUT << "Error@DeleteDirectory(): It is not a directory." << ENDL;
        return false;
    }

    const std::string pathStr( path );

    const size_t numAsterisks = Count( pathStr, '*' );

    if( numAsterisks > 0 )
    {
        COUT << "Error@DeleteDirectory(): You cannot use * in the path for safety." << ENDL;
        return false;
    }

    std::string cmd( "rm -rf " );
    cmd += path;

    system( cmd.c_str() );

    return true;
}

bool DeleteFile( const char* file )
{
    if( IsDirectory( file ) == true )
    {
        COUT << "Error@DeleteFile(): It is not a file." << ENDL;
        return false;
    }

    std::string cmd( "rm -f " );
    cmd += file;

    system( cmd.c_str() );

    return true;
}

std::string CurrentPath()
{
    char cdir[512];
    getcwd( cdir, 512 );
    return std::string( cdir );
}

std::string FileExtension( const char* path )
{
    std::string fileStr( path );

    StringArray tokens;
    tokens.setByTokenizing( fileStr, "." );

    return tokens.last();
}

std::string RemoveExtension( const char* path )
{
    std::string pathStr( path );

    const int lastIndex = LastIndexOf( pathStr, '.' );

    return SubString( pathStr, 0, lastIndex );
}

std::string ChangeSeparators( const char* path )
{
    std::string pathStr( path );

    const size_t N = pathStr.length();

    for( size_t i=0; i<N; ++i )
    {
        char& c = pathStr[i];

        if( c == '/' )       { c = '\\'; }
        else if( c == '\\' ) { c = '/';  }
    }

    return pathStr;
}

bool GetFileList( const char* path, std::vector<std::string>& files, bool asFullPath )
{
    files.clear();

    DIR* dp;
    struct dirent* dirp;

    if( !( dp = opendir( path ) ) )
    {
        COUT << "Error@GetFileList(): Failed open file " << path << ENDL;
        return false;
    }

    while( ( dirp = readdir(dp) ) != NULL )
    {
        const std::string shortPath = dirp->d_name;

        std::string fullPath = path;
        fullPath += "/" + shortPath;

        if( IsDirectory( fullPath.c_str() ) ) { continue; }

        if( asFullPath )
        {
            files.push_back( fullPath );
        }
        else
        {
            files.push_back( shortPath );
        }
    }

    closedir( dp );

	return true;
}

bool GetFileList( const char* path, const std::string& extension, std::vector<std::string>& files, bool asFullPath )
{
    std::vector<std::string> candidates;

    if( !GetFileList( path, candidates, asFullPath ) )
    {
        return false;
    }

	const size_t numFiles = candidates.size();

    for( size_t i=0; i<numFiles; ++i )
	{
        const std::string ext = FileExtension( candidates[i].c_str() );

		if( ext == extension )
		{
			files.push_back( candidates[i] );
		}
	}

	return true;
}

bool GetDirectoryList( const char* path, std::vector<std::string>& directories, bool asFullPath )
{
    directories.clear();

    DIR* dp;
    struct dirent* dirp;

    if( !( dp = opendir( path ) ) )
    {
        COUT << "Error@GetDirectoryList(): Failed open file " << path << ENDL;
        return false;
    }

    while( ( dirp = readdir(dp) ) != NULL )
    {
        const std::string shortPath = dirp->d_name;

        if( shortPath == "." ) { continue; }
        if( shortPath == ".." ) { continue; }

        std::string fullPath = path;
        fullPath += "/" + shortPath;

        if( IsDirectory( fullPath.c_str() ) )
        {
            if( asFullPath )
            {
                directories.push_back( fullPath );
            }
            else
            {
                directories.push_back( shortPath );
            }
        }
    }

    closedir( dp );

	return true;
}

void Write( const std::string& str, std::ofstream& fout )
{
    const int n = (int)str.size();
    fout.write( (char*)&n, sizeof(int) );

    if( n )
    {
        fout.write( (char*)&str[0], n*sizeof(char) );
    }
}

void Read( std::string& str, std::ifstream& fin )
{
    str.clear();

    int n = 0;
    fin.read( (char*)&n, sizeof(int) );

    if( n )
    {
        str.resize( n );
        fin.read( (char*)&str[0], n*sizeof(char) );
    }
}

bool Save( const std::string& str, const char* filePathName )
{
    const size_t n = str.size();

    std::ofstream fout( filePathName, std::ios::out );

    if( fout.fail() || !fout.is_open() )
    {
        COUT << "Error@Save(): Failed to save file: " << filePathName << ENDL;
        return false;
    }

    fout.write( (char*)&str[0], n*sizeof(char) );

    fout.close();

    return true;
}

bool Load( std::string& str, const char* filePathName )
{
    if( DoesFileExist( filePathName ) == false )
    {
        COUT << "Error@Load(): Invalid file path & name." << ENDL;
        return false;
    }

    std::ifstream fin( filePathName, std::ios::in );

    if( fin.fail() )
    {
        COUT << "Error@Load(): Failed to load file." << ENDL;
        return false;
    }

    std::stringstream buffer;
    buffer << fin.rdbuf();

    str = buffer.str();

    fin.close();

    return true;
}

BORA_NAMESPACE_END

