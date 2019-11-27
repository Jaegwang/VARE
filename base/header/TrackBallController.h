//-----------------------//
// TrackBallController.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
// last update: 2018.01.05                               //
//-------------------------------------------------------//

#ifndef _BoraTrackBallController_h_
#define _BoraTrackBallController_h_

#include <Bora.h>
#include <GLFW/glfw3.h>

BORA_NAMESPACE_BEGIN

class TrackBallController
{
    private:
	
	    glm::fvec3 position = glm::fvec3(0.f, 0.f, 0.f);
        glm::fquat rotation = glm::fquat(1.f, 0.f, 0.f, 0.f);

        bool pressed[3] = {false,false,false};
        float mpx, mpy;

        float width, height;

    public:

        TrackBallController()
        {}
        ~TrackBallController()
        {}

        void setFramebufferSize( const int w, const int h )
        {
            width = (float)w;
            height = (float)h;
        }

        void cursorPosCallback( float xpos, float ypos )
        {
            const float dx = (xpos - mpx)/width*2.f;
            const float dy = (ypos - mpy)/height*2.f;

            if( pressed[0] == true ) // left button -> rotate
            {
                glm::fvec3 angular_vector = glm::fvec3(1.f,0.f,0.f)*dy + glm::fvec3(0.f,1.f,0.f)*dx;
                const float magnitude = glm::length(angular_vector);

                glm::fquat q;
                
                if( magnitude > 1e-10f )
                {
                    q.w = Cos(magnitude*0.5f);
                    
                    glm::fvec3 axis = (Sin(0.5f*magnitude)/magnitude)*angular_vector;
                    
                    q.x = axis.x; 
                    q.y = axis.y; 
                    q.z = axis.z;
                }

                rotation =  q*rotation; 
                rotation = glm::normalize(rotation);
            }

            if( pressed[1] == true )
            {
            }

            if( pressed[2] == true )
            {
            }

            mpx = xpos;
            mpy = ypos;            
        }

        void mouseButtonCallback( int button, int action, int mods )
        {
            if( action == GLFW_PRESS )
            {
                if( button == GLFW_MOUSE_BUTTON_LEFT ) pressed[0] = true;
                else if( button == GLFW_MOUSE_BUTTON_MIDDLE ) pressed[1] = true;
                else if( button == GLFW_MOUSE_BUTTON_RIGHT ) pressed[2] = true;
            }
            else if( action == GLFW_RELEASE )
            {
                if( button == GLFW_MOUSE_BUTTON_LEFT ) pressed[0] = false;
                else if( button == GLFW_MOUSE_BUTTON_MIDDLE ) pressed[1] = false;
                else if( button == GLFW_MOUSE_BUTTON_RIGHT ) pressed[2] = false;
            }
        }

        void scrollCallback( float xoffset, float yoffset )
        {
            position.z += yoffset;
        }
        
        const float getTranslation( glm::fvec3& trans )
        {
            trans = position;
            return glm::length( position );
        }

        const float getRotation( glm::fvec3& axis )
        {
            axis.x = rotation.x;
            axis.y = rotation.y;
            axis.z = rotation.z;                        
            return ACos( rotation.w ) * 360.f/3.141592f;
        }

};

BORA_NAMESPACE_END

#endif

