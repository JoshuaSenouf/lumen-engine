#ifndef OBJECT_H
#define OBJECT_H

#include <cuda_runtime.h>

#include "material.h"


struct RayObject
{
        float3 origin;
        float3 direction;

        __device__ RayObject(float3 tempOrigin, float3 tempDirection) : origin(tempOrigin), direction(tempDirection)
        {

        }
};


struct SphereObject
{
        float radius;

        float3 position;
        float3 color;
        float3 emissiveColor;

        materialType material;
};


#endif // OBJECT_H
