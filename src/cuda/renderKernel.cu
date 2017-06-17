#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <vector_types.h>

#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"

#include "renderKernel.h"
#include "cutil_math.h"
#include "object.h"
#include "material.h"

#define PI 3.14159265359f
#define FOV_ANGLE 0.5135f
#define EPSILON 0.03f


inline float clamp(float x)
{
    return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x;
}


inline int hdrToSGRB(float x)
{
    return int(pow(clamp(x), 1 / 2.2) * 255);
}


__device__ inline unsigned int WangHash(unsigned int seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed = seed + (seed << 3);
    seed = seed ^ (seed >> 4);
    seed = seed * 0x27d4eb2d;
    seed = seed ^ (seed >> 15);

    return seed;
}


__device__ static float getrandom(unsigned int *seed0, unsigned int *seed1) // random number function from Samuel Lapere
{
    *seed0 = 36969 * ((*seed0) & 65535) + ((*seed0) >> 16);
    *seed1 = 18000 * ((*seed1) & 65535) + ((*seed1) >> 16);

    unsigned int ires = ((*seed0) << 16) + (*seed1);

    union
    {
            float f;
            unsigned int ui;
    } res;

    res.ui = (ires & 0x007fffff) | 0x40000000;

    return (res.f - 2.0f) / 2.0f;
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


__device__ bool checkSceneIntersect(RayObject &ray, SphereObject* spheresList, float &closestSphereDist, int &closestSphereID)
{
    float d;
    float inf = closestSphereDist = 1e20;

    for(int i = 0; i < (sizeof(spheresList) + 1); i++) // Eeewwwwwww
    {
        if((d = checkSphereIntersect(spheresList[i], ray)) && d < closestSphereDist)
        {
            closestSphereDist = d;
            closestSphereID = i;
        }
    }

    return closestSphereDist < inf;
}


__host__ __device__ float3 computeCosineWeightedImportanceSampling(float3 localW, float3 localU, float3 localV, float rand1, float rand2, float sqrtRand2)
{
    return normalize(localU * cos(rand1) * sqrtRand2 + localV * sin(rand1) * sqrtRand2 + localW * sqrtf(1 - rand2));

}


__host__ __device__ float3 computePerfectlyReflectedRay(float3 rayDirection, float3 intersectionNormal)
{
    return rayDirection - 2.0f * intersectionNormal * dot(intersectionNormal, rayDirection);
}


__device__ float3 computeRadiance(RayObject &ray, SphereObject* spheresList, int lightBounces, unsigned int *seed1, unsigned int *seed2)
{
    float3 colorAccumulation = make_float3(0.0f, 0.0f, 0.0f);
    float3 colorMask = make_float3(1.0f, 1.0f, 1.0f);

    for (int bounces = 0; bounces < lightBounces; bounces++)
    {
        float closestSphereDist;
        int closestSphereID = 0;

        if (!checkSceneIntersect(ray, spheresList, closestSphereDist, closestSphereID))
            return make_float3(0.0f, 0.0f, 0.0f);

        const SphereObject &hitSphere = spheresList[closestSphereID];
        float3 hitCoord = ray.origin + ray.direction * closestSphereDist;
        float3 hitNormal = normalize(hitCoord - hitSphere.position);
        float3 hitFrontNormal = dot(hitNormal, ray.direction) < 0.0f ? hitNormal : hitNormal * -1.0f;

        colorAccumulation += colorMask * hitSphere.emissiveColor;

        float random1 = 2.0f * M_PI * getrandom(seed1, seed2);
        float random2 = getrandom(seed1, seed2);
        float random2Square = sqrtf(random2);

        float3 nextRayDir;

        // DIFFUSE
        if (hitSphere.material == 1)
        {
            float3 localOrthoW = hitFrontNormal;
            float3 localOrthoU = normalize(cross((fabs(localOrthoW.x) > 0.1f ? make_float3(0.0f, 1.0f, 0.0f) : make_float3(1.0f, 0.0f, 0.0f)), localOrthoW));
            float3 localOrthoV = cross(localOrthoW, localOrthoU);

            // Cosine Weighted Importance Sampling
            nextRayDir = computeCosineWeightedImportanceSampling(localOrthoW, localOrthoU, localOrthoV, random1, random2, random2Square);
            hitCoord += hitFrontNormal * EPSILON;

            colorMask *= dot(nextRayDir, hitFrontNormal);
        }

        // (PERFECT) SPECULAR
        else if (hitSphere.material == 2)
        {
            nextRayDir = computePerfectlyReflectedRay(ray.direction, hitNormal);
            hitCoord += hitFrontNormal * EPSILON;
        }

        // REFRACT
        else if (hitSphere.material == 3)
        {

        }

        ray.direction = nextRayDir;
        ray.origin = hitCoord;

        colorMask *= hitSphere.color;
        colorMask *= 2.0f;
    }

    return colorAccumulation;
}


__global__ void renderDispatcher(float3 *dataHost, int renderWidth, int renderHeight, int sampleCount, int lightBounces, int sphereCount, SphereObject *spheresList)
{

    unsigned int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int pixelY = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int pixelIndex = (renderHeight - pixelY - 1) * renderWidth + pixelX;

    unsigned int seed1 = pixelX;
    unsigned int seed2 = pixelY;

    RayObject cameraRay(make_float3(50.0f, 52.0f, 295.6f), normalize(make_float3(0.0f, -0.042612f, -1.0f)));
    float3 rayOffsetX = make_float3(renderWidth * FOV_ANGLE / renderHeight, 0.0f, 0.0f);
    float3 rayOffsetY = normalize(cross(rayOffsetX, cameraRay.direction)) * FOV_ANGLE;

    float3 pixelColor;
    pixelColor = make_float3(0.0f);

    for (int sample = 0; sample < sampleCount; sample++)
    {
        float3 primaryRay = cameraRay.direction + rayOffsetX * ((0.25f + pixelX) / renderWidth - 0.5f) + rayOffsetY * ((0.25f + pixelY) / renderHeight - 0.5f);
        RayObject tempRay(cameraRay.origin + primaryRay * 40.0f, normalize(primaryRay));

        pixelColor = pixelColor + computeRadiance(tempRay, spheresList, lightBounces, &seed1, &seed2) * (1.0f / sampleCount);
    }

    dataHost[pixelIndex] = make_float3(clamp(pixelColor.x, 0.0f, 1.0f), clamp(pixelColor.y, 0.0f, 1.0f), clamp(pixelColor.z, 0.0f, 1.0f));
}


extern "C"
void lumenRender(int renderWidth, int renderHeight, int samples, int lightBounces, int sphereCount, SphereObject *spheresList)
{
    printf("RENDER CONFIG :\n\n"
           "Width = %d\n"
           "Height = %d\n"
           "Samples = %d\n"
           "Bounces = %d\n"
           "Sphere count : %d\n\n", renderWidth, renderHeight, samples, lightBounces, sphereCount);

    float3* dataHost = new float3[renderWidth * renderHeight];
    float3* dataDevice;

    cudaMalloc(&dataDevice, renderWidth * renderHeight * sizeof(float3));

    dim3 cudaThreadsBlock(8, 8);
    dim3 cudaBlocksGrid(renderWidth / cudaThreadsBlock.x, renderHeight / cudaThreadsBlock.y);

    printf("LOG : CUDA is ready !\n");

    renderDispatcher <<< cudaBlocksGrid, cudaThreadsBlock >>>(dataDevice,renderWidth, renderHeight, samples, lightBounces, sphereCount, spheresList);

    cudaMemcpy(dataHost, dataDevice, renderWidth * renderHeight * sizeof(float3), cudaMemcpyDeviceToHost);
    cudaFree(dataDevice);

    printf("LOG : Render done !\n");

    FILE *f = fopen("lumenRender.ppm", "w");
    fprintf(f, "P3\n%d %d\n%d\n", renderWidth, renderHeight, 255);

    for (int i = 0; i < renderWidth*renderHeight; i++)
    {
        fprintf(f, "%d %d %d ", hdrToSGRB(dataHost[i].x), hdrToSGRB(dataHost[i].y), hdrToSGRB(dataHost[i].z));
    }

    fclose(f);

    printf("LOG : Render successfuly saved in lumenRender.ppm !");

    delete[] dataHost;

    printf("\n\n//////////////////////\n\n");
}
