#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <vector_types.h>
#include "device_launch_parameters.h"
#include "cutil_math.h"
#include "renderKernel.h"

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


__device__ SphereObject spheres[] =
{
    {1e5f, {1e5f + 1.0f, 40.8f, 81.6f}, {0.0f, 0.0f, 0.0f}, {0.75f, 0.25f, 0.25f}, DIFFUSE}, // Left wall
    {1e5f, {-1e5f + 99.0f, 40.8f, 81.6f}, {0.0f, 0.0f, 0.0f}, {.25f, .25f, .75f}, DIFFUSE}, // Right wall
    {1e5f, {50.0f, 40.8f, 1e5f}, {0.0f, 0.0f, 0.0f}, {.75f, .75f, .75f}, DIFFUSE}, // Back wall
    {1e5f, {50.0f, 40.8f, -1e5f + 600.0f}, {0.0f, 0.0f, 0.0f}, {1.00f, 1.00f, 1.00f}, DIFFUSE}, // Front wall
    {1e5f, {50.0f, 1e5f, 81.6f}, {0.0f, 0.0f, 0.0f}, {.75f, .75f, .75f}, DIFFUSE}, // Floor
    {1e5f, {50.0f, -1e5f + 81.6f, 81.6f}, {0.0f, 0.0f, 0.0f}, {.75f, .75f, .75f}, DIFFUSE}, // Roof
    {16.5f, {27.0f, 16.5f, 47.0f}, {0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, DIFFUSE}, // Sphere 1
    {16.5f, {73.0f, 16.5f, 78.0f}, {0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, DIFFUSE}, // Sphere 2
    {600.0f, {50.0f, 681.6f - .77f, 81.6f}, {2.0f, 1.8f, 1.6f}, {0.0f, 0.0f, 0.0f}, DIFFUSE}  // Roof light
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


__device__ bool checkSceneIntersect(RayObject &ray, float &t, int &id)
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


void lumenRender(int renderWidth, int renderHeight, int samples, int lightBounces)
{
    printf("\nRENDER CONFIG :\n\n"
           "Width = %d\n"
           "Height = %d\n"
           "Samples = %d\n"
           "Bounces = %d\n\n", renderWidth, renderHeight, samples, lightBounces);

    float3* dataHost = new float3[renderWidth * renderHeight];
    float3* dataDevice;

    cudaMalloc(&dataDevice, renderWidth * renderHeight * sizeof(float3));

    dim3 cudaThreadsBlock(16, 16);
    dim3 cudaBlocksGrid(renderWidth / cudaThreadsBlock.x, renderHeight / cudaThreadsBlock.y);


    printf("LOG : CUDA is ready !");

    printf("\n\n//////////////////////\n\n");
}
