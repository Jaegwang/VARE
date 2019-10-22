
#pragma once
#include <VARE.h>

VARE_NAMESPACE_BEGIN

class FieldOP
{
    public:
    
        static void gradient( const ScalarDenseField& src, VectorDenseField& dst );
        static void gradient( const ScalarSparseField& src, VectorSparseField& dst );

        static void normalGradient( const ScalarSparseField& src, VectorSparseField& dst );

        static void divergence( const VectorDenseField& src, ScalarDenseField& dst );
        static void divergence( const VectorSparseField& src, ScalarSparseField& dst );

        static void curvature( const ScalarDenseField& surf, ScalarDenseField& curv );
        static void curvature( const ScalarSparseField& surf, ScalarSparseField& curv );

        static void curvature( const VectorDenseField& norm, ScalarDenseField& curv );
        static void curvature( const VectorSparseField& norm, ScalarSparseField& curv );

        static void curl( const VectorDenseField& vel, VectorDenseField& cur );
        static void curl( const VectorSparseField& vel, VectorSparseField& cur );

        static void normalCurl( const VectorSparseField& vel, VectorSparseField& cur );
};

VARE_NAMESPACE_END

