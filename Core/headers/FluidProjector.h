
#pragma once
#include <VARE.h>

VARE_NAMESPACE_BEGIN

class FluidProjector
{
    private:

        SparseMatrix<float> _A;

        FloatArray _M;
        FloatArray _x, _b;

        IndexArray _fluidCellBuffer;

    public:

        float wallLevel=-1e+30f;
        int   maxIteration=150;
        //bool  usePreconditioner=false;

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

        FluidProjector();

        // for Dense Field
        void buildLinearSystem( DenseField<size_t>& typField, const VectorDenseField& velField, const ScalarDenseField& _pressureField );
        int  solve( VectorDenseField& velField, ScalarDenseField& _pressureField, DenseField<size_t>& typField );

        // for Sparse Field
        void buildLinearSystem( IdxSparseField& typField, const Vec3fSparseField& veloField, const FloatSparseField& pressField );
        int  solve( Vec3fSparseField& veloField, FloatSparseField& pressField, const IdxSparseField& typField );

        // for debugging
        void printLinearSystem();
};

VARE_NAMESPACE_END
