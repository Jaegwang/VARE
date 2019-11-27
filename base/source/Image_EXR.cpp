//---------------//
// Image_EXR.cpp //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.03.23                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

bool Image::saveEXR( const char* filePathName ) const
{
    const int& w = _width;
    const int& h = _height;

    const std::string filePathNameStr( filePathName ); // for safety
    Imf::RgbaOutputFile file( filePathNameStr.c_str(), w, h, Imf::WRITE_RGBA );

    file.setFrameBuffer( _pixels, 1, w );

    file.writePixels( h );

    return true;
}

bool Image::loadEXR( const char* filePathName )
{
    const std::string filePathNameStr( filePathName ); // for safety
    Imf::RgbaInputFile file( filePathNameStr.c_str() );

    const Imath::Box2i dw = file.dataWindow();

    const int& w = _width  = dw.max.x - dw.min.x + 1;
    const int& h = _height = dw.max.y - dw.min.y + 1;

    Image::allocate();

    file.setFrameBuffer( _pixels - dw.min.x - dw.min.y * w, 1, w );
    file.readPixels( dw.min.y, dw.max.y );

    switch( file.compression() )
    {
        default:
        case Imf::NO_COMPRESSION:    { _compression = "NONE";  break; }
        case Imf::RLE_COMPRESSION:   { _compression = "RLD";   break; }
        case Imf::ZIPS_COMPRESSION:  { _compression = "ZIPS";  break; }
        case Imf::ZIP_COMPRESSION:   { _compression = "ZIP";   break; }
        case Imf::PIZ_COMPRESSION:   { _compression = "PIZ";   break; }
        case Imf::PXR24_COMPRESSION: { _compression = "PXR24"; break; }
        case Imf::B44_COMPRESSION:   { _compression = "B44";   break; }
        case Imf::B44A_COMPRESSION:  { _compression = "B44A";  break; }
        case Imf::DWAA_COMPRESSION:  { _compression = "DWAA";  break; }
        case Imf::DWAB_COMPRESSION:  { _compression = "DWAB";  break; }
    }

    return true;
}

BORA_NAMESPACE_END

