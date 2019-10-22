
#pragma once
#include <VARE.h>

VARE_NAMESPACE_BEGIN

class Particles
{
	public:

        CharArray   type, state;
        UIntArray   flags;
        IndexArray  uniqueId, index;
        FloatArray  radius, mass, density, adjacency, distance, age, lifespan, curvature, vorticity, velocityNormal;
        Vec3fArray  position, velocity, curl, acceleration, force, angularMomentum, normal, color, textureCoordinates;
};

VARE_NAMESPACE_END

