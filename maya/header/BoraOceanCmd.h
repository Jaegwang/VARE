//----------------//
// BoraOceanCmd.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.03.09                               //
//-------------------------------------------------------//

#ifndef _BoraOceanCmd_h_
#define _BoraOceanCmd_h_

#include <Bora.h>
#include <MayaCommon.h>
#include <MayaUtils.h>
#include <BoraOcean.h>

class BoraOceanCmd : public MPxCommand
{
	public:

		static MString name;

	public:

		virtual MStatus doIt( const MArgList& );
		virtual bool isUndoable() const { return false; }

		static void *creator() { return new BoraOceanCmd; }
		static MSyntax newSyntax();

	private:

		MString getNodeName   ( const MArgDatabase& argData );
		MString getToolName   ( const MArgDatabase& argData );
		MString getFilePath   ( const MArgDatabase& argData );
		MString getFileName   ( const MArgDatabase& argData );
        int     getStartFrame ( const MArgDatabase& argData );
        int     getEndFrame   ( const MArgDatabase& argData );

        void exportData( MString nodeName, MString filePath, MString fileName, int startFrame, int endFrame );
        void importData( MString nodeName, MString filePath, MString fileName );
};

#endif

