//-------------------//
// FluidProjection.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.07.02                               //
//-------------------------------------------------------//

#pragma once
#include <Bora.h>

BORA_NAMESPACE_BEGIN

class FluidProjection
{
    private:

        SparseMatrix<float> _A;

        FloatArray _M;
        FloatArray _x, _b;

        IndexArray _fluidCellBuffer;

    public:

        float wallLevel=-1e+30f;
        int   maxIteration=150;
        bool  usePreconditioner=false;

    public:

        enum CELLTYPE
        {
            kAir=0,
            kSolid,
            kWall,
            kMark,
            kFluid /* Fluid Cell >= kFluid */
        };

    public:

        FluidProjection();

        // for Dense Field
        void buildLinearSystem( DenseField<size_t>& typField, const VectorDenseField& velField, const ScalarDenseField& _pressureField );
        int  solve( VectorDenseField& velField, ScalarDenseField& _pressureField, DenseField<size_t>& typField );

        // for Sparse Field
        void buildLinearSystem( IdxSparseField& typField, const Vec3fSparseField& veloField, const FloatSparseField& pressField );
        int  solve( Vec3fSparseField& veloField, FloatSparseField& pressField, const IdxSparseField& typField );

        // for debugging
        void printLinearSystem();
};

BORA_NAMESPACE_END
