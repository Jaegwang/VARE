//--------------------//
// OceanGLSLShaders.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.04.02                               //
//-------------------------------------------------------//

#ifndef _BoraOceanGLSLShaders_h_
#define _BoraOceanGLSLShaders_h_

#include <Bora.h>

BORA_NAMESPACE_BEGIN

///////////////////
// Vertex Shader //
static const char* OceanVS =
R"(
#version 430 core

// uniform input data
uniform mat4 projectionMatrix;
uniform mat4 modelViewMatrix;

// per vertex input data
layout(location=0) in vec4  vPOS;
layout(location=1) in vec3  vNRM;
layout(location=2) in float vWAV;
layout(location=3) in float vCRS;

// output to the fragment shader
out vec3  fPOS;
out vec3  fNRM;
out float fWAV;
out float fCRS;

void main()
{
	gl_Position = projectionMatrix * modelViewMatrix * vPOS;

    fPOS = gl_Position.xyz;
    fNRM = vNRM;
    fWAV = vWAV;
    fCRS = vCRS;
}

)";

/////////////////////
// Fragment Shader //
static const char* OceanFS =
R"(
#version 430 core

// uniform input data
uniform vec3      cameraPosition;
uniform vec3      deepWaterColor;
uniform vec3      shallowWaterColor;
uniform float     minHeight;
uniform float     maxHeight;
uniform float     glossiness;
uniform float     exposure;
uniform int       hasSkyTexture;
uniform sampler2D skyTexture;
uniform int       showTangles;
uniform vec3      tangleColor;

// per fragment input data
layout(location=0) in vec3  fPOS;
layout(location=1) in vec3  fNRM;
layout(location=3) in float fWAV;
layout(location=4) in float fCRS;

out vec4 outColor;

vec2 Spherical_to_ST( vec3 normal, vec3 position )
{
	vec3 u = normalize( position );
	vec3 r = reflect( u, normal );

	float m = 2.0 * sqrt( r.x*r.x + r.y*r.y + (r.z+1.0)*(r.z+1.0) );

	return vec2( r.x/m+0.5, r.y/m+0.5 );
}

void main()
{
    vec3 P = fPOS;
    vec3 N = fNRM;
    vec3 E = normalize( cameraPosition - P );

    float F = 1.0 - dot( E, N );
    float fresnel = pow( F, 2.0 );

    float foamColor = clamp( fCRS, 0.0, 1.0 );

    vec3 specularColor;

    if( hasSkyTexture == 1 )
    {
        vec2 st = Spherical_to_ST( N, P );

        float s = st.s;
        float t = st.t;

        vec4 skyColor = texture2D( skyTexture, vec2(s,-t) );
        specularColor = skyColor.rgb * glossiness;
    }

    float heightMixer = clamp( (fWAV-minHeight)/(maxHeight-minHeight), 0.0, 1.0 );

    float sssAmount = clamp( pow( F, 2.0 ) * pow( heightMixer, 5.0 ) * 2.0, 0.0, 1.0 );
    vec3 sssOceanColor = mix( deepWaterColor, shallowWaterColor, heightMixer );
    vec3 sssColor = mix( sssOceanColor, shallowWaterColor+vec3(0.1,0.1,0.1), sssAmount );

    float crest = clamp( fCRS, 0.0, 1.0 );
    vec3 crestColor = vec3( crest, crest, crest );

    vec3 finalColor = ( specularColor * fresnel ) + sssColor + crestColor;

    float bright = 1.0;
    float Y = dot( vec3( 0.30, 0.59, 0.11 ), finalColor );
    float YD = exposure * ( exposure / bright + 1.0 ) / ( exposure + 1.0 );
    finalColor *= YD;

    if( ( showTangles == 1 ) && ( fNRM.y < 0.0 ) )
    {
        outColor = vec4( tangleColor.xyz, 1.0 );
    }
    else
    {
        outColor = vec4( finalColor, 1.0 );
    }
}
)";

BORA_NAMESPACE_END

#endif

