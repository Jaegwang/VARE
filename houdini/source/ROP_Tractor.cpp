//-----------------//
// ROP_Tractor.cpp //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2019.04.10                               //
//-------------------------------------------------------//

#include <ROP_Tractor.h>

const char* ROP_Tractor::uniqueName = "boratractor";
const char* ROP_Tractor::labelName = "Bora Tractor";
const char* ROP_Tractor::menuName = "Bora";

PRM_Template ROP_Tractor::nodeParameters[] = 
{
    houSeperatorParameter( "submit_sep", "submit_sep" ),    
    houSimpleButtonParameter( "submit", "Submit", ROP_Tractor::submitCallback ),
    houMenuParameter ( "server", "Servers", 3, "10.0.0.25", "10.0.0.30", "10.0.0.106" ),

    PRM_Template()
};

int
ROP_Tractor::submitCallback( void* data, int index, float time, const PRM_Template* tplate )
{
    OP_Context context( time );

    ROP_Tractor* node = (ROP_Tractor*)data;

    const int trange = node->evalInt( "trange" , 0, 0 );
    const int frame_start = node->evalInt( "f", 0, 0);
    const int frame_end = node->evalInt( "f", 1, 0);
    const int frame_inc = node->evalInt( "f", 2, 0);

    std::stringstream frame_range;
    frame_range << "-e -f " << frame_start << " " << frame_end << " -i " << frame_inc;

    /*
    import tractor.api.author as author;

    author.setEngineClientParam(hostname="10.0.0.25", port=80);

    job = author.Job( title="test", priority=200, service="GPUFARM");
    job.newTask( title="A task", argv=["hrender", "-h"], service="GPUFARM");

    job.spool();

    author.closeEngineClient();
    */
    
    /*
    int nInputs = node->nConnectedInputs();
    for(int i=0; i<nInputs; ++i )
    {
        int ix = node->getNthConnectedInput( i );
        OP_Node* connNode = node->getInput( ix );


        std::cout<< connNode->getFullPath() << std::endl;
        std::cout<< connNode->getName() << std::endl;

        UT_String val;
        connNode->evalString( val, "source", 0, 0.f );


        std::stringstream ss;
        ss << "hrender -d " << val << " " << frame_range.str() << " " << std::getenv("HIPFILE") << " " << "&";

        std::cout<< ss.str() << std::endl;
        
        system( ss.str().c_str() );

    }
    */ 

    //system("hrender -d /out/fetch1 /home/jaegwang.lim/Desktop/test_out.hip & ");

    return 1;
}

OP_Node*
ROP_Tractor::constructor( OP_Network* net, const char* name, OP_Operator* entry )
{
    return new ROP_Tractor( net, name, entry );
}

ROP_Tractor::ROP_Tractor( OP_Network* net, const char* name, OP_Operator* entry )
: ROP_Node( net, name, entry )
{
}

ROP_Tractor::~ROP_Tractor()
{    
}

int
ROP_Tractor::startRender( int nframes, fpreal s, fpreal e )
{
    int rcode = 1;
    return rcode;
}

void cookConnectedNodes( fpreal time, OP_Node* node )
{
    if( node == 0 ) return;

    int nInputs = node->nConnectedInputs();
    for(int i=0; i<nInputs; ++i )
    {
        int ix = node->getNthConnectedInput( i );
        OP_Node* connNode = node->getInput( ix );

        cookConnectedNodes( time, connNode );
    }  

    /* do something */
    {
        std::cout<< node->getName() << std::endl;

        UT_String fullpath;
        node->getFullPath( fullpath );

        std::cout<< fullpath << std::endl;
    }
}

ROP_RENDER_CODE
ROP_Tractor::renderFrame( fpreal time, UT_Interrupt* boss )
{
    return ROP_CONTINUE_RENDER;
}

ROP_RENDER_CODE
ROP_Tractor::endRender()
{
    return ROP_CONTINUE_RENDER;
}

