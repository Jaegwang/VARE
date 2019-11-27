//-------------//
// FileUtils.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.03.09                               //
//-------------------------------------------------------//

#ifndef _BoraFileUtils_h_
#define _BoraFileUtils_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

bool IsDirectory( const char* path );

bool IsSymbolicLink( const char* path );

bool DoesFileExist( const char* path );

size_t FileSize( const char* path );

bool CreateDirectory( const char* path, const mode_t permission=0755 );

bool DeleteDirectory( const char* path );

bool DeleteFile( const char* file );

std::string CurrentPath();

std::string FileExtension( const char* path );

std::string RemoveExtension( const char* path );

std::string ChangeSeparators( const char* path );

bool GetFileList( const char* path, std::vector<std::string>& files, bool asFullPath=false );

bool GetFileList( const char* path, const std::string& extension, std::vector<std::string>& files, bool asFullPath=false );

bool GetDirectoryList( const char* path, std::vector<std::string>& directories, bool asFullPath=false );

void Write( const std::string& str, std::ofstream& fout );

void Read( std::string& str, std::ifstream& fin );

bool Save( const std::string& str, const char* filePathName );

bool Load( std::string& str, const char* filePathName );

BORA_NAMESPACE_END

#endif

