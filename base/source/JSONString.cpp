//----------------//
// JSONString.cpp //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2018.03.09                               //
//-------------------------------------------------------//

#include <Bora.h>
#include <boost/foreach.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

BORA_NAMESPACE_BEGIN

JSONString::JSONString()
{
    // nothing to do
}

void JSONString::clear()
{
	intData.clear();
	floatData.clear();
	vectorData.clear();
	stringData.clear();
}

void JSONString::append( const char* name, const int& value )
{
	intData[ name ] = value;
}

void JSONString::append( const char* name, const float& value )
{
	floatData[ name ] = value;
}

void JSONString::append( const char* name, const double& value )
{
	floatData[ name ] = (float)value;
}

void JSONString::append( const char* name, const Vec3f& value )
{
	vectorData[ name ] = value;
}

void JSONString::append( const char* name, const std::string& value )
{
	stringData[ name ] = value;
}

bool JSONString::get( const char* name, int& value )
{
	std::map<std::string,int>::const_iterator itr = intData.find( name );
	if( itr == intData.end() ) { return false; }
	value = itr->second;
	return true;
}

bool JSONString::get( const char* name, float& value )
{
	std::map<std::string,float>::const_iterator itr = floatData.find( name );
	if( itr == floatData.end() ) { return false; }
	value = itr->second;
	return true;
}

bool JSONString::get( const char* name, double& value )
{
	std::map<std::string,float>::const_iterator itr = floatData.find( name );
	if( itr == floatData.end() ) { return false; }
	value = (double)itr->second;
	return true;
}

bool JSONString::get( const char* name, Vec3f& value )
{	
	std::map<std::string,Vec3f>::const_iterator itr = vectorData.find( name );
	if( itr == vectorData.end() ) { return false; }
	value = itr->second;
	return true;
}

bool JSONString::get( const char* name, std::string& value )
{
	std::map<std::string,std::string>::const_iterator itr = stringData.find( name );
	if( itr == stringData.end() ) { return false; }
	value = itr->second;
	return true;
}

size_t JSONString::numItems() const
{
    size_t count = 0;

    count += intData.size();
    count += floatData.size();
    count += vectorData.size();
    count += stringData.size();

    return count;
}

std::string JSONString::json() const
{
    std::string str;
    JSONString::get( str );

    return str;
}

void JSONString::get( std::string& str ) const
{
	boost::property_tree::ptree root;

	for( std::map<std::string,int>::const_iterator it=intData.begin(); it!=intData.end(); ++it )
	{
		std::string name = std::string("int.") + it->first;
		root.put( name.c_str(), it->second );
	}

	for( std::map<std::string,float>::const_iterator it=floatData.begin(); it!=floatData.end(); ++it )
	{
		std::string name = std::string("float.") + it->first;
		root.put( name.c_str(), it->second );
	}

	for( std::map<std::string,Vec3f>::const_iterator it=vectorData.begin(); it!=vectorData.end(); ++it )
	{
		std::string name = std::string("vector.") + it->first;

		boost::property_tree::ptree row;
		for( int i=0; i<3; ++i )
		{
			boost::property_tree::ptree cell;
			cell.put_value( (it->second)[i] );

			row.push_back( std::make_pair( "", cell ) );
		}

		root.add_child( name.c_str(), row );
    }

	for( std::map<std::string,std::string>::const_iterator it=stringData.begin(); it!=stringData.end(); ++it )
	{	
		std::string name = std::string("string.") + it->first;
		root.put( name.c_str(), (it->second).c_str() );
	}

    std::stringstream ss;
	boost::property_tree::write_json( ss, root );

	str = ss.str();
}

void JSONString::set( const std::string& str )
{
    JSONString::clear();

	std::stringstream ss;
	ss << str;

	boost::property_tree::ptree props;
    boost::property_tree::read_json( ss, props );

	if( props.count( "int" ) > 0 )
	for( boost::property_tree::ptree::value_type& vt : props.get_child( "int" ) )
	{
		intData[ vt.first ] = vt.second.get_value<int>();
	}
	
	if( props.count( "float" ) > 0 )
	for( boost::property_tree::ptree::value_type& vt : props.get_child( "float" ) )
	{
		floatData[ vt.first ] = vt.second.get_value<float>();		
	}

	if( props.count( "string" ) > 0 )
	for( boost::property_tree::ptree::value_type& vt : props.get_child( "string" ) )
	{
		stringData[ vt.first ] = vt.second.get_value<std::string>();
	}

	if( props.count( "vector" ) > 0 )
	for( boost::property_tree::ptree::value_type& vt : props.get_child( "vector" ) )
	{
		Vec3f v;
		
		int ix(0);
		BOOST_FOREACH( auto &sub, vt.second )
		{
			v[ix++] = sub.second.get_value<float>();
		}

		vectorData[vt.first] = v;
	}
}

bool JSONString::save( const char* filePathName ) const
{
    std::string str;

    JSONString::get( str );

    return Save( str, filePathName );
}

bool JSONString::load( const char* filePathName )
{
    JSONString::clear();

    std::string str;

    if( !Load( str, filePathName ) ) { return false; }

    JSONString::set( str );

    return true;
}

BORA_NAMESPACE_END

