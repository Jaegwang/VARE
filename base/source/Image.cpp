//-----------//
// Image.cpp //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.04.23                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

Image::Image()
{
    // nothing to do
}

Image::~Image()
{
    Image::clear();
}

void Image::clear()
{
    _width = _height = _size = _numChannels = 0;

    _compression = "none";

    Image::release();
}

void Image::allocate()
{
    const int n = _width * _height;

    if( _size != n )
    {
        Image::release();

        _pixels = new Pixel[ _size = n ];
    }

    memset( (char*)_pixels, 0, sizeof(Pixel)*n );
}

void Image::release()
{
    if( _pixels )
    {
        delete[] _pixels;
    }

    _pixels = nullptr;
}

bool Image::create( const int& width, const int& height )
{
    if( ( width < 1 ) || ( height < 1 ) )
    {
        COUT << "Error@Iamge::create(): Invalid dimension." << ENDL;
        return false;
    }

    _width  = width;
    _height = height;

    Image::allocate();

    return true;
}

void Image::zeroize()
{
    const int n = Image::numPixels();
    if( n == 0 ) { return; }

    memset( (char*)_pixels, 0, sizeof(Pixel)*n );
}

bool Image::save( const char* filePathName ) const
{
    const std::string ext = FileExtension( filePathName );

    if( ext == "exr" ) { return Image::saveEXR( filePathName ); }
    if( ext == "jpg" ) { return Image::saveJPG( filePathName ); }
    if( ext == "hdr" ) { return Image::saveHDR( filePathName ); }

    COUT << "Error@Image::save(): Not supported file format." << ENDL;

    return false;
}

bool Image::load( const char* filePathName )
{
    if( DoesFileExist( filePathName ) == false )
    {
        COUT << "Error@Iamge::load(): Invalid file path & name." << ENDL;
        return false;
    }

    const std::string ext = FileExtension( filePathName );

    if( ext == "exr" ) { return Image::loadEXR( filePathName ); }
    if( ext == "jpg" ) { return Image::loadJPG( filePathName ); }
    if( ext == "hdr" ) { return Image::loadHDR( filePathName ); }

    COUT << "Error@Image::load(): Not supported file format." << ENDL;

    return false;
}

BORA_NAMESPACE_END

