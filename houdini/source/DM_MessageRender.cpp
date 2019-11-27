//----------------------//
// DM_MessageRender.cpp //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2019.04.03                               //
//-------------------------------------------------------//

#include <DM_MessageRender.h>

std::map< size_t, DM_MessageRender::Data > DM_MessageRender::map;

void DM_MessageRender::renderStringStream( RE_Render* r, const DM_SceneHookData &hook_data, std::stringstream& ss )
{
    const RE_Font& defaultFont = RE_Render::getViewportFont();
    const FONT_Info& info = defaultFont.getFontInfo();

    RE_Font& font = RE_Font::get( info, 20.f );
    UT_Color text_col = UT_RED;

    int fHeight = font.getHeight();
    int h = fHeight;

    r->pushColor( text_col );
    r->setFont( font );            

    std::string str;
    while( std::getline( ss, str ) )
    {
        UT_String tt( str );
        int w = font.getStringWidth( tt );

        r->textMoveW( (hook_data.view_width-w)*0.5f, hook_data.view_height-h-10 );
        r->putString( tt );

        h += fHeight;
    }

    r->popColor();
}

