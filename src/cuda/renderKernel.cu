#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <vector_types.h>
#include "device_launch_parameters.h"
#include "renderKernel.h"
#include "cutil_math.h"

#define PI 3.14159265359


enum materialType
{
    DIFFUSE,
    SPECULAR,
    REFRACT
};


struct RayObject
{
        float3 origin;
        float3 direction;
};


struct SphereObject
{
        float radius;

        float3 position;
        float3 emissiveColor;
        float3 color;

        materialType material;
};



inline float clamp(float x)
{
    return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x;
}


inline int hdrToSGRB(float x)
{
    return int(pow(clamp(x), 1 / 2.2) * 255);
}


__device__ float checkSphereIntersect(SphereObject sphere, RayObject ray)
{
    float3 op = sphere.position - ray.origin;
    float t, eps = 1e-4;
    float b = dot(op, ray.direction);

    float quadDis = (b * b) - dot(op, op) + (sphere.radius * sphere.radius);

    if(quadDis < 0)
        return 0;
    else
        quadDis = sqrt(quadDis);

    return (t = b - quadDis) >  eps ? t : ((t = b + quadDis) > eps ? t : 0);
}


__device__ bool checkSceneIntersect(RayObject &ray, SphereObject *spheres,float &t, int &id)
{
    float d;
    float inf = t = 1e20;

    for(int i = 0; i < (sizeof(spheres)/sizeof(SphereObject)); i++)
    {
        if((d = checkSphereIntersect(spheres[i], ray)) && d < t)
        {
            t = d;
            id = i;
        }
    }

    return t < inf;
}


__device__ float computeRadiance()
{

}

__global__ void renderDispatcher(float3 *dataHost, int renderWidth, int renderHeight, int samples, int lightBounces)
{

}


void lumenRender(int renderWidth, int renderHeight, int samples, int lightBounces, int sphereCount, SphereObject *spheres)
{
    printf("\nRENDER CONFIG :\n\n"
           "Width = %d\n"
           "Height = %d\n"
           "Samples = %d\n"
           "Bounces = %d\n"
           "Sphere count : %d\n"
           "Sphere 1 radius : %f\n\n", renderWidth, renderHeight, samples, lightBounces, sphereCount, spheres[1].radius);

    float3* dataHost = new float3[renderWidth * renderHeight];
    float3* dataDevice;

    cudaMalloc(&dataDevice, renderWidth * renderHeight * sizeof(float3));

    dim3 cudaThreadsBlock(16, 16);
    dim3 cudaBlocksGrid(renderWidth / cudaThreadsBlock.x, renderHeight / cudaThreadsBlock.y);


    printf("LOG : CUDA is ready !");

    printf("\n\n//////////////////////\n\n");
}
