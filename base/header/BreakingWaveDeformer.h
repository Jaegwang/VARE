//------------------------//
// BreakingWaveDeformer.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.12.18                               //
//-------------------------------------------------------//

#pragma once
#include <Bora.h>

BORA_NAMESPACE_BEGIN

class BreakingWaveDeformer
{
    private:

        Array< SplineCurve* > _curves;

    public:

        BreakingWaveDeformer() { _curves.initialize( 0, kUnified ); }
        ~BreakingWaveDeformer() { clear(); }

        SplineCurve* addControlCurve();
        SplineCurve* getControlCurve( const size_t n );

        void resize( const size_t n );
        void clear();

        void wave( PointArray& output, const PointArray& points, const float rad );
};

BORA_NAMESPACE_END

