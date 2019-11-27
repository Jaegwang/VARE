//----------------------//
// DM_SimInfoRender.cpp //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2019.04.03                               //
//-------------------------------------------------------//

#include <DM_SimInfoRender.h>

std::map< size_t, DM_SimInfoRender::Data > DM_SimInfoRender::map;

void DM_SimInfoRender::renderStringStream( RE_Render* r, const DM_SceneHookData &hook_data, std::stringstream& ss )
{
    const RE_Font& defaultFont = RE_Render::getViewportFont();
    const FONT_Info& info = defaultFont.getFontInfo();

    RE_Font& font = RE_Font::get( info, 11.f );
    UT_Color text_col = UT_YELLOW;
    //UT_Color text_col = hook_data.disp_options->common().defaultWireColor();

    int fHeight = font.getHeight();
    int h = fHeight;

    r->pushColor( text_col );
    r->setFont( font );            

    std::string str;
    while( std::getline( ss, str ) )
    {
        UT_String tt( str );

        r->textMoveW( 10, hook_data.view_height-h-10 );
        r->putString( tt );

        h += fHeight;
    }

    r->popColor();
}

