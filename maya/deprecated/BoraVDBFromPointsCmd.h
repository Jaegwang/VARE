//------------------------//
// BoraVDBFromPointsCmd.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2017.09.20                               //
//-------------------------------------------------------//

#ifndef _BoraVDBFromPointsCmd_h_
#define _BoraVDBFromPointsCmd_h_

#include <MayaCommon.h>

class BoraVDBFromPointsCmd : public MPxCommand
{
    public:

        static const MString name;

        BoraVDBFromPointsCmd() {};
        virtual ~BoraVDBFromPointsCmd() {};

        virtual MStatus doIt( const MArgList& args );
		virtual bool isUndoable() const { return false; }

        MStatus undoIt() { return MS::kSuccess; }     

        static void* creator() { return new BoraVDBFromPointsCmd(); }
        static MSyntax newSyntax() { MSyntax syntax; return syntax; }
        
};

#endif

