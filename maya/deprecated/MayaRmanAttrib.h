//------------------//
// BoraRmanAttrib.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2017.09.20                               //
//-------------------------------------------------------//

#ifndef _BoraRmanAttrib_h_
#define _BoraRmanAttrib_h_

#include <MayaCommon.h>

inline void
CreateRmanAttrib( const char* nodeName, const char* attribName, const char* value )
{
    int check(0);
    MGlobal::executeCommand( "exists rmanGetAttrName", check );

    if( check > 0 )
    {   
        std::stringstream ss;
        ss << "rmanAddAttr " << nodeName << " `rmanGetAttrName " << attribName << "` " << '\"' << value << '\"';

        std::stringstream ss1;
        ss1 << "rmanSetAttr " << nodeName << " `rmanGetAttrName " << attribName << "` " << '\"' << value << '\"';

        MGlobal::executeCommand( ss.str().c_str() );
        MGlobal::executeCommand( ss1.str().c_str() );
    }
}

inline void
CreateRmanAttrib( const char* nodeName, const char* attribName, const int& value )
{
    int check(0);
    MGlobal::executeCommand( "exists rmanGetAttrName", check );

    if( check > 0 )
    {   
        std::stringstream ss;
        ss << "rmanAddAttr " << nodeName << " `rmanGetAttrName " << attribName << "` " << value;
        
        std::stringstream ss1;        
        ss1 << "rmanSetAttr " << nodeName << " `rmanGetAttrName " << attribName << "` " << value;
        
        MGlobal::executeCommand( ss.str().c_str() );
        MGlobal::executeCommand( ss1.str().c_str() );        
    }
}

inline void
CreateRmanAttrib( const char* nodeName, const char* attribName, const float& value )
{
    int check(0);
    MGlobal::executeCommand( "exists rmanGetAttrName", check );

    if( check > 0 )
    {   
        std::stringstream ss;
        ss << "rmanAddAttr " << nodeName << " `rmanGetAttrName " << attribName << "` " << value;

        std::stringstream ss1;
        ss1 << "rmanSetAttr " << nodeName << " `rmanGetAttrName " << attribName << "` " << value;
        
        MGlobal::executeCommand( ss.str().c_str() );
        MGlobal::executeCommand( ss1.str().c_str() );        
    }
}

#endif

