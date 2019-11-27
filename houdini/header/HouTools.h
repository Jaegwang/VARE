//------------//
// HouTools.h //
//-------------------------------------------------------//
// author: Jaegwawng Lim @ Dexter Studios                //
// last update: 2019.04.03                               //
//-------------------------------------------------------//

#pragma once

#include <HouCommon.h>

/* Create houdini-node parameters */
inline PRM_Template houFloatParameter( const char* name, const char* label, const float defValue )
{
    PRM_Name* prm_name = new PRM_Name( name, label );
    PRM_Default* prm_default = new PRM_Default[1]{ defValue };

    return PRM_Template( PRM_FLT, 1, prm_name, prm_default );
}

inline PRM_Template houVec2fParameter( const char* name, const char* label, const Vec2f defValue )
{
    PRM_Name* prm_name = new PRM_Name( name, label );
    PRM_Default* prm_default = new PRM_Default[2]{ defValue.x, defValue.y };

    return PRM_Template( PRM_XYZ, 2, prm_name, prm_default );
}

inline PRM_Template houVec3fParameter( const char* name, const char* label, const Vec3f defValue )
{
    PRM_Name* prm_name = new PRM_Name( name, label );
    PRM_Default* prm_default = new PRM_Default[3]{ defValue.x, defValue.y, defValue.z };

    return PRM_Template( PRM_XYZ, 3, prm_name, prm_default );
}

inline PRM_Template houVec4fParameter( const char* name, const char* label, const Vec4f defValue )
{
    PRM_Name* prm_name = new PRM_Name( name, label );
    PRM_Default* prm_default = new PRM_Default[4]{ defValue.x, defValue.y, defValue.z, defValue.w };

    return PRM_Template( PRM_XYZ, 4, prm_name, prm_default );
}

inline PRM_Template houIntParameter( const char* name, const char* label, const int defValue )
{
    PRM_Name* prm_name = new PRM_Name( name, label );
    PRM_Default* prm_default = new PRM_Default[1]{ defValue };

    return PRM_Template( PRM_INT, 1, prm_name, prm_default );
}

inline PRM_Template houVec2iParameter( const char* name, const char* label, const Vec2i defValue )
{
    PRM_Name* prm_name = new PRM_Name( name, label );
    PRM_Default* prm_default = new PRM_Default[2]{ defValue.x, defValue.y };

    return PRM_Template( PRM_INT, 2, prm_name, prm_default );
}

inline PRM_Template houVec3iParameter( const char* name, const char* label, const Vec3i defValue )
{
    PRM_Name* prm_name = new PRM_Name( name, label );
    PRM_Default* prm_default = new PRM_Default[3]{ defValue.x, defValue.y, defValue.z };

    return PRM_Template( PRM_INT, 3, prm_name, prm_default );
}

inline PRM_Template houBoolParameter( const char* name, const char* label, const bool defValue )
{
    PRM_Name* prm_name = new PRM_Name( name, label );
    PRM_Default* prm_default = new PRM_Default[1]{ defValue };

    return PRM_Template( PRM_TOGGLE, 1, prm_name, prm_default );
}

inline PRM_Template houRGBParameter( const char* name, const char* label, const Vec3f defColor )
{
    PRM_Name* prm_name = new PRM_Name( name, label );
    PRM_Default* prm_default = new PRM_Default[3]{ defColor.r, defColor.g, defColor.b };

    return PRM_Template( PRM_RGB, 3, prm_name, prm_default );
}

inline PRM_Template houSeperatorParameter( const char* name, const char* label )
{
    PRM_Name* prm_name = new PRM_Name( name, label );
    
    return PRM_Template( PRM_SEPARATOR, 1, prm_name, 0 );
}

inline PRM_Template houGeoFileParameter( const char* name, const char* label )
{
    PRM_Name* prm_name = new PRM_Name( name, label );
    
    return PRM_Template( PRM_GEOFILE, 1, prm_name, 0 );
}

inline PRM_Template houPicFileParameter( const char* name, const char* label )
{
    PRM_Name* prm_name = new PRM_Name( name, label );
    
    return PRM_Template( PRM_PICFILE, 1, prm_name, 0 );
}

inline PRM_Template houSwitcherParameter( const char* name, const char* label, const int num, ... )
{
    va_list ap;
    va_start( ap, num );

    PRM_Name* prm_name = new PRM_Name( name, label );

    PRM_Default* lists = new PRM_Default[ num ];

    for( int i=0; i<num; ++i )
    {
        char* name = va_arg( ap, char* );
        int n = va_arg( ap, int );
        lists[i] = PRM_Default( n, name );
    }

    va_end( ap );
    return PRM_Template( PRM_SWITCHER, num, prm_name, lists );
}

inline PRM_Template houSimpleButtonParameter( const char* name, const char* label, PRM_Callback callback )
{
    PRM_Name* prm_name = new PRM_Name( name, label );
    return PRM_Template( PRM_CALLBACK, 1, prm_name, 0, 0, 0, callback );
}

inline PRM_Template houLabelParameter( const char* name, const char* label )
{
    PRM_Name* prm_name = new PRM_Name( name, label );
    return PRM_Template( PRM_LABEL, 4, prm_name );
}

/* Evaluate houdini-node parameters */
inline int houEvalInt( OP_Node* node, const char* name )
{
    return node->evalInt( name, 0, 0 );
}

inline float houEvalFloat( OP_Node* node, const char* name )
{
    return node->evalFloat( name, 0, 0.f );
}

inline Vec2f houEvalVec2f( OP_Node* node, const char* name )
{
    return Vec2f( node->evalFloat( name, 0, 0.f ),
                  node->evalFloat( name, 1, 0.f ));
}

inline Vec2i houEvalVec2i( OP_Node* node, const char* name )
{
    return Vec2i( node->evalInt( name, 0, 0 ),
                  node->evalInt( name, 1, 0 ));
}

inline Vec3f houEvalVec3f( OP_Node* node, const char* name )
{
    return Vec3f( node->evalFloat( name, 0, 0.f ),
                  node->evalFloat( name, 1, 0.f ),
                  node->evalFloat( name, 2, 0.f ));
}

inline Vec3i houEvalVec3i( OP_Node* node, const char* name )
{
    return Vec3i( node->evalInt( name, 0, 0 ),
                  node->evalInt( name, 1, 0 ),
                  node->evalInt( name, 2, 0 ));
}

inline bool houEvalBool( OP_Node* node, const char* name )
{
    return (bool)(node->evalFloat( name, 0, 0.f ));
}

inline Vec3f houEvalRGB( OP_Node* node, const char* name )
{
    return Vec3f( node->evalFloat( name, 0, 0.f ),
                  node->evalFloat( name, 1, 0.f ),
                  node->evalFloat( name, 2, 0.f ));
}

inline const UT_String houEvalString( OP_Node* node, const char* name )
{
    UT_String val;
    node->evalString( val, name, 0, 0.f );
    return val;
}

inline PRM_Template houMenuParameter( const char* name, const char* label, const int num, ... )
{
    va_list ap;
    va_start( ap, num );

    PRM_Name* prm_name = new PRM_Name( name, label );

    PRM_Name* choices = new PRM_Name[ num+1 ];

    for( int i=0; i<num; ++i )
    {
        char* name = va_arg( ap, char* );
        choices[i] = PRM_Name( name, name );
    }

    choices[num] = PRM_Name(0);

    va_end( ap );

    PRM_ChoiceList* menu = new PRM_ChoiceList( PRM_CHOICELIST_SINGLE, choices );
    
    return PRM_Template( PRM_ORD, 1, prm_name, 0, menu );
}

