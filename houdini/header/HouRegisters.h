//----------------//
// HouRegisters.h //
//-------------------------------------------------------//
// author: Jaegwawng Lim @ Dexter Studios                //
// last update: 2019.04.10                               //
//-------------------------------------------------------//

#pragma once

#include <HouCommon.h>

/* Houdini node register functions */
#define REGISTER_SOP(table,node,min_input,max_input,icon_name) { OP_Operator* sop = new OP_Operator(\
                                                                       node::uniqueName,    \
                                                                       node::labelName,     \
                                                                       node::constructor,   \
                                                                       node::nodeParameters,\
                                                                       min_input,           \
                                                                       max_input,           \
                                                                       0,0,                 \
                                                                       node::inputLabels,   \
                                                                       1,                   \
                                                                       node::menuName );    \
                                                       if( icon_name ) sop->setIconName(icon_name);\
                                                       table->addOperator( sop ); };


#define REGISTER_DOP(table,node,min_input,max_input,icon_name) { DOP_Operator* dop = new DOP_Operator(\
                                                                       node::uniqueName,     \
                                                                       node::labelName,      \
                                                                       node::constructor,    \
                                                                       node::nodeParameters, \
                                                                       node::menuName,       \
                                                                       min_input,            \
                                                                       max_input,            \
                                                                       0,0,                  \
                                                                       1 );                  \
                                                       if( icon_name ) dop->setIconName(icon_name);\
                                                       dop->setOpTabSubMenuPath( node::menuName );\
                                                       table->addOperator( dop ); };


#define REGISTER_ROP(table,node,min_input,max_input) { OP_Operator* rop = new OP_Operator(                       \
                                                                node::uniqueName,                                \
                                                                node::labelName,                                 \
                                                                node::constructor,                               \
                                                                new OP_TemplatePair( node::getROPbaseTemplate(), \
                                                                new OP_TemplatePair( node::nodeParameters ) ),   \
                                                                min_input, max_input,                            \
                                                                new OP_VariablePair( node::myVariableList ),     \
                                                                OP_FLAG_GENERATOR );                             \
                                                        table->addOperator( rop ); };


#define REGISTER_VEX(node) { new VEX_VexOp(\
					                        node::signature, \
                                            node::compute,   \
                                            VEX_ALL_CONTEXT, \
                                            node::init,      \
                                            node::clear,     \
                                            VEX_OPTIMIZE_2,  \
                                            true ); };

