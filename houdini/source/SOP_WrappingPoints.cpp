//------------------------//
// SOP_WrappingPoints.cpp //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.09.10                               //
//-------------------------------------------------------//

#include <SOP_WrappingPoints.h>

const char* SOP_WrappingPoints::uniqueName = "borawrappingpoints";
const char* SOP_WrappingPoints::labelName = "Bora Wrapping Points";
const char* SOP_WrappingPoints::menuName = "Bora";

const char* SOP_WrappingPoints::inputLabels[] =
{
    "Source Points"
};

PRM_Template SOP_WrappingPoints::nodeParameters[] =
{
    houPicFileParameter( "texture", "Texture" ),
    PRM_Template()
};

OP_Node*
SOP_WrappingPoints::constructor( OP_Network*net, const char *name, OP_Operator* entry )
{
    return new SOP_WrappingPoints( net, name, entry );
}

SOP_WrappingPoints::SOP_WrappingPoints( OP_Network*net, const char* name, OP_Operator* entry )
: SOP_Node(net, name, entry)
{
}

SOP_WrappingPoints::~SOP_WrappingPoints()
{
}

OP_ERROR
SOP_WrappingPoints::cookMySop( OP_Context &context )
{
    OP_AutoLockInputs inputs( this );
    if (inputs.lock(context) >= UT_ERROR_ABORT) return error();

    duplicateSource(0, context);

    GA_ROHandleV3 rohUvw = gdp->findFloatTuple( GA_ATTRIB_POINT, "uvw", 3 );

    if( rohUvw.isValid() )
    {
        UT_String image_path = houEvalString( this, "texture" );

        if( _image.load( image_path.c_str() ) )
        {
            const int width = _image.width();
            const int height = _image.height();

            size_t pointnum = gdp->getNumPoints();
            size_t startoff = gdp->pointOffset( 0 );

            _uvwArray.initialize( pointnum );
            rohUvw.getBlock( startoff, pointnum, (UT_Vector3*)_uvwArray.pointer() );


            auto kernel = [=] BORA_HOST ( const size_t n )
            {
                Vec3f& uvw = _uvwArray[n];
                const float i = uvw.i*(float)(width-1);
                const float j = uvw.k*(float)(height-1);

                uvw = _image.lerp( i,j );
            };

            LaunchCudaHost( kernel, 0, _uvwArray.size() );

            GA_RWHandleV3 rwhCd = GA_RWHandleV3( gdp->addFloatTuple( GA_ATTRIB_POINT, "Cd", 3 ) );

            rwhCd.setBlock( startoff, pointnum, (UT_Vector3*)_uvwArray.pointer() );

        }
    }

    return error();
}
   
