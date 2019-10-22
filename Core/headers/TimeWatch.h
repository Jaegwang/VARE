
#pragma once
#include <VARE.h>

VARE_NAMESPACE_BEGIN

class TimeWatch
{   
    private:

        struct Task
        {
            std::string title;
            int depth=0;

            std::clock_t clock;
        };

        std::stack< Task > _tasks;
        std::stack< Task > _records;

        std::stringstream _ss;

    public:

        void start( const char* name );
        void stop();

        std::string printString()
        {
            return _ss.str();
        }
};

inline void TimeWatch::start( const char* title )
{
    if( _tasks.empty() ) _ss.str("");

    Task task;
    task.title = std::string( title );
    task.clock = std::clock();
    task.depth = _tasks.size();

    _tasks.push( task );
}

inline void TimeWatch::stop()
{
    if( _tasks.empty() ) return;

    Task task = _tasks.top();
    _tasks.pop();

    task.clock = std::clock() - task.clock;
    _records.push( task );

    if( _tasks.empty() == true )
    {
        while( !_records.empty() )
        {
            Task task = _records.top();
            _records.pop();

            for( int n=0; n<task.depth+1; ++n ) _ss << '#';
            _ss << " " << task.title << " : ";
            _ss << (float)task.clock / CLOCKS_PER_SEC << " sec";
            _ss << std::endl;
        }
    }
}

VARE_NAMESPACE_END

