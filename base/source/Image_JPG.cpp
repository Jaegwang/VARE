//---------------//
// Image_JPG.cpp //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.03.23                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

bool Image::saveJPG( const char* filePathName ) const
{
    FILE* fp = fopen( filePathName, "wb" );

    if( !fp )
    {
        COUT << "Error@saveJPG(): Failed to open " << filePathName << ENDL;
        return false;
    }

    jpeg_compress_struct cinfo;
    jpeg_error_mgr jerr;

    JSAMPLE* samples = new JSAMPLE[_width*3];

    cinfo.err = jpeg_std_error( &jerr );

    jpeg_create_compress( &cinfo );
    jpeg_stdio_dest( &cinfo, fp );

    cinfo.image_width      = _width;
    cinfo.image_height     = _height;
    cinfo.input_components = 3;
    cinfo.in_color_space   = JCS_RGB;

    jpeg_set_defaults( &cinfo );

    const int quality = 95;
    jpeg_set_quality( &cinfo, quality, (boolean)true );

    jpeg_start_compress( &cinfo, (boolean)true );

    JSAMPROW row_pointer = samples;

    for( int j=0; j<_height; ++j )
    {
        for( int i=0; i<_width; ++i )
        {
            const Pixel& p = _pixels[ Image::index( i, j ) ];

            samples[3*i  ] = (JSAMPLE)( p.r * 255.f );
            samples[3*i+1] = (JSAMPLE)( p.g * 255.f );
            samples[3*i+2] = (JSAMPLE)( p.b * 255.f );
        }

        jpeg_write_scanlines( &cinfo, &row_pointer, 1 );
    }

    jpeg_finish_compress( &cinfo );
    jpeg_destroy_compress( &cinfo );

    delete[] samples;

    fclose( fp );

    return true;
}

bool Image::loadJPG( const char* filePathName )
{
    FILE* fp = fopen( filePathName, "rb" );

    if( !fp )
    {
        COUT << "Error@loadJPG(): Failed to open " << filePathName << ENDL;
        return false;
    }

    jpeg_decompress_struct cinfo;
    jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error( &jerr );

    jpeg_create_decompress( &cinfo );
    jpeg_stdio_src( &cinfo, fp );

    jpeg_read_header( &cinfo, (boolean)true );
    jpeg_start_decompress( &cinfo );

    _width       = cinfo.output_width;
    _height      = cinfo.output_height;
    _numChannels = 3;
    _compression = "NONE";

    Image::allocate();

    JSAMPLE* samples = new JSAMPLE[_width*_numChannels];

    for( int j=0; j<_height; ++j )
    {
        jpeg_read_scanlines( &cinfo, &samples, 1 );

        for( int i=0; i<_width; ++i )
        {
            Pixel& p = _pixels[ Image::index( i, j ) ];

            p.r = (half)( samples[3*i  ] / 255.f );
            p.g = (half)( samples[3*i+1] / 255.f );
            p.b = (half)( samples[3*i+2] / 255.f );
            p.a = (half)( 1.f );
        }
    }

    delete[] samples;

    jpeg_finish_decompress( &cinfo );
    jpeg_destroy_decompress( &cinfo );

    fclose( fp );

    return true;
}

BORA_NAMESPACE_END

