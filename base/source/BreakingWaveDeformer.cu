//-------------------------//
// BreakingWaveDeformer.cu //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.12.18                               //
//-------------------------------------------------------//

#include <Bora.h>

BORA_NAMESPACE_BEGIN

SplineCurve* BreakingWaveDeformer::addControlCurve()
{
    SplineCurve* curve(0);
    cudaMallocManaged( &curve, sizeof(SplineCurve) );

    _curves.append( curve );
        
    return curve;
}

SplineCurve* BreakingWaveDeformer::getControlCurve( const size_t n )
{
    if( _curves.size() <= n ) return 0;

    if( _curves[n] == 0 )
    {
        SplineCurve* curve(0);
        cudaMallocManaged( &curve, sizeof(SplineCurve) );
        
        _curves[n] = curve;
        return curve;
    }

    return _curves[n];
}

void BreakingWaveDeformer::resize( const size_t n )
{
    _curves.resize( n );
}

void BreakingWaveDeformer::clear()
{
    for( int n=0; n<_curves.size(); ++n )
    {
        cudaFree( _curves[n] );
    }

    _curves.clear();
}

void BreakingWaveDeformer::wave( PointArray& output, const PointArray& points, const float rad )
{
    output.initialize( points.size(), kUnified );

    Vec3f* pos = points.pointer();
    Vec3f* out = output.pointer();

    const Array< SplineCurve* >& curves = _curves;

    auto kernel_update = [=] BORA_DEVICE ( const size_t n )
    {
        Vec3f disp(0.f);
        float weight(0.f);

        for( int c=0; c<curves.size(); ++c )
        {
            const SplineCurve& curve = *(curves[c]);
            const int ncp = curve.cp.size();

            Vec3f p0 = curve.begin();
            Vec3f p1 = curve.end();
            p0.y = p1.y = pos[n].y;

            Vec3f cen = (p0+p1)*0.5f;
            const float R = p0.distanceTo( p1 )*0.5f;

            const double prm = Clamp( ((p1-p0)*(pos[n]-p0))/((p1-p0)*(p1-p0)), 0.f ,1.f );

            const Vec3f q = p0*(1.f-prm) + p1*prm;
            const Vec3f t = curve.point( prm*(double)(ncp-1) );

            float u = Min( (pos[n]-q).length()/rad, 1.f );

            const float v = 1.f - u*u*u;
            const float k_tri = 70.f/81.f * v*v*v;

            Vec3f cliff = (t-q) * k_tri;

            const float w = 1.f-u;

            disp += cliff * w;
            weight += w;
        }

        out[n] = pos[n];
        if( weight > 1e-10f ) out[n] += disp/weight;
    };
    

    LaunchCudaDevice( kernel_update, 0, points.size() );
    SyncCuda();    
}

BORA_NAMESPACE_END

