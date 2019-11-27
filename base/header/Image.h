//---------//
// Image.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.04.23                               //
//-------------------------------------------------------//

#ifndef _BoraImage_h_
#define _BoraImage_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

typedef Imf::Rgba Pixel;
// size(Imf::Rgba) = 8 (8 = 2 bytes x 4 channels)

class Image
{
    private:

        int _width       = 0; // the image width
        int _height      = 0; // the image height
        int _size        = 0; // # of pixels (= width * height)
        int _numChannels = 0; // the # of channels

        std::string _compression = "none";

        Pixel* _pixels = nullptr; // the pixel data

    public:

        Image();

        ~Image();

        void clear();

        bool create( const int& width, const int& height );

        void zeroize();

        bool save( const char* filePathName ) const;

        bool load( const char* filePathName );

    public: // inline functions

        int width() const
        {
            return _width;
        }

        int height() const
        {
            return _height;
        }

        int numPixels() const
        {
            return _size;
        }

        const char* compression() const
        {
            return _compression.c_str();
        }

        int index( const int& i, const int& j ) const
        {
            return ( i + _width * j );
        }

        Pixel& operator[]( const int& i )
        {
            return _pixels[i];
        }

        const Pixel& operator[]( const int& i ) const
        {
            return _pixels[i];
        }

        Pixel& operator()( const int& i, const int& j )
        {
            return _pixels[ i + _width * j ];
        }

        const Pixel& operator()( const int& i, const int& j ) const
        {
            return _pixels[ i + _width * j ];
        }

        const Vec3f lerp( const float& i, const float& j ) const;

        Pixel* pointer() const
        {
            return _pixels;
        }

    private:

        void allocate();
        void release();

        bool saveEXR( const char* filePathName ) const;
        bool loadEXR( const char* filePathName );

        bool saveJPG( const char* filePathName ) const;
        bool loadJPG( const char* filePathName );

        bool saveHDR( const char* filePathName ) const;
        bool loadHDR( const char* filePathName );
};


inline const Vec3f Image::lerp( const float& _i, const float& _j ) const
{
    const size_t i = Clamp( int(_i-0.5f), 0, (int)_width-2 );
    const size_t j = Clamp( int(_j-0.5f), 0, (int)_height-2 );

    const Vec2f corner( (float)i+0.5f, (float)j+0.5f );

    const float x = Clamp( (_i - corner.x), 0.f, 1.f );
    const float y = Clamp( (_j - corner.y), 0.f, 1.f );

    const Image& f = (*this);

    const Pixel& p00 = f(i  , j  );
    const Pixel& p10 = f(i+1, j  );
    const Pixel& p01 = f(i  , j+1);
    const Pixel& p11 = f(i+1, j+1);

    const Vec3f v00( p00.r, p00.g, p00.b );
    const Vec3f v10( p10.r, p10.g, p10.b );
    const Vec3f v01( p01.r, p01.g, p01.b );
    const Vec3f v11( p11.r, p11.g, p11.b );

    const Vec3f c00 = v00*(1.f-x) + v10*x;
    const Vec3f c11 = v01*(1.f-x) + v11*x;

    return c00*(1.f-y) + c11*y;
}

BORA_NAMESPACE_END

#endif

