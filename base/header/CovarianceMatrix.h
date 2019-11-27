//--------------------//
// CovarianceMatrix.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2017.10.26                               //
//-------------------------------------------------------//

#ifndef _BoraCovarianceMatrix_h_
#define _BoraCovarianceMatrix_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

inline Matrix33<float>
CovarianceMatrixFromPoints( const Vec3fArray& points, const Vec3f& centroid )
{
    Matrix33<float> mat(0.f);

    for( int i=0; i<3; ++i )
    for( int j=0; j<3; ++j )
    {
        float cov(0.f);
        for( int n=0; n<points.num(); ++n )
        {
            cov += (centroid[i]-points[n][i]) * (centroid[j]-points[n][j]);
        }

        mat[i][j] = cov / (float)(points.num()-1);
    }

    return mat;
}

inline Matrix33<float>
CovarianceMatrixFromPoints( const std::vector<Vec3f>& points, const Vec3f& centroid )
{
    Matrix33<float> mat(0.f);

    for( int i=0; i<3; ++i )
    for( int j=0; j<3; ++j )
    {
        float cov(0.f);
        for( int n=0; n<points.size(); ++n )
        {
            cov += (centroid[i]-points[n][i]) * (centroid[j]-points[n][j]);
        }

        mat[i][j] = cov / (float)(points.size()-1);
    }

    return mat;   
}

inline Matrix33<float>
CovarianceMatrixFromPoints( const Vec3fArray& points )
{
    Vec3f center(0.f, 0.f, 0.f);
    for( int n=0; n<points.num(); ++n )
    {
        center += points[n];
    }
    center /= (float)points.num();
    
    return CovarianceMatrixFromPoints( points, center );
}

inline Matrix33<float>
CovarianceMatrixFromPoints( const std::vector<Vec3f>& points )
{
    Vec3f center(0.f, 0.f, 0.f);
    for( int n=0; n<points.size(); ++n )
    {
        center += points[n];
    }
    center /= (float)points.size();

    return CovarianceMatrixFromPoints( points, center );
}

BORA_NAMESPACE_END

#endif

