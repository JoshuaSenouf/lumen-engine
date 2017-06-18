#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <vector_types.h>

#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"
#include "cutil_math.h"

#include "renderKernel.h"
#include "material.h"
#include "object.h"

#define PI 3.14159265359f
#define FOV_ANGLE 0.5135f
#define EPSILON 0.03f


union GLColor  // Allow us to convert the pixel value to one that OpenGL can read and use
{
	float colorValue;
	uchar4 colorComponents;
};


__device__ inline float clamp(float x)
{
    return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x;
}


__device__ inline int hdrToSGRB(float x)
{
    return int(powf(clamp(x), 1 / 2.2) * 255);
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


__device__ bool checkSceneIntersect(RayObject &ray, int sphereCount, SphereObject* spheresList, float &closestSphereDist, int &closestSphereID)
{
    float d;
    float inf = closestSphereDist = 1e20;

    for(int i = 0; i < sphereCount; i++)
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


__device__ float3 computeRadiance(RayObject &ray, int sphereCount, SphereObject* spheresList, int lightBounces, curandState *cudaRNG)
{
    float3 colorAccumulation = make_float3(0.0f, 0.0f, 0.0f);
    float3 colorMask = make_float3(1.0f, 1.0f, 1.0f);

    for (int bounces = 0; bounces < lightBounces; bounces++)
    {
        float closestSphereDist;
        int closestSphereID = 0;

        if (!checkSceneIntersect(ray, sphereCount, spheresList, closestSphereDist, closestSphereID))
            return make_float3(0.0f, 0.0f, 0.0f);

        const SphereObject &hitSphere = spheresList[closestSphereID];
        float3 hitCoord = ray.origin + ray.direction * closestSphereDist;
        float3 hitNormal = normalize(hitCoord - hitSphere.position);
        float3 hitFrontNormal = dot(hitNormal, ray.direction) < 0.0f ? hitNormal : hitNormal * -1.0f;

        colorAccumulation += colorMask * hitSphere.emissiveColor;

        float random1 = 2.0f * PI * curand_uniform(cudaRNG);
        float random2 = curand_uniform(cudaRNG);
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


__global__ void renderDispatcher(float3 *dataHost, float3* accumBuffer, int renderWidth, int renderHeight,
								int sampleCount, int lightBounces, int sphereCount, SphereObject *spheresList,
								int frameCounter)
{
    unsigned int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int pixelY = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int pixelIndex = (renderHeight - pixelY - 1) * renderWidth + pixelX;

    int threadIndex = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    curandState cudaRNG;
    curand_init(WangHash(frameCounter) + threadIndex, 0, 0, &cudaRNG); // We create a new seed using curand and our framecounter

    RayObject cameraRay(make_float3(50.0f, 52.0f, 295.6f), normalize(make_float3(0.0f, -0.042612f, -1.0f)));
    float3 rayOffsetX = make_float3(renderWidth * FOV_ANGLE / renderHeight, 0.0f, 0.0f);
    float3 rayOffsetY = normalize(cross(rayOffsetX, cameraRay.direction)) * FOV_ANGLE;

    float3 pixelColor;
    pixelColor = make_float3(0.0f);

    for (int sample = 0; sample < sampleCount; sample++)
    {
        float3 primaryRay = cameraRay.direction + rayOffsetX * ((0.25f + pixelX) / renderWidth - 0.5f) + rayOffsetY * ((0.25f + pixelY) / renderHeight - 0.5f);
        RayObject tempRay(cameraRay.origin + primaryRay * 40.0f, normalize(primaryRay));

        pixelColor = pixelColor + computeRadiance(tempRay, sphereCount, spheresList, lightBounces, &cudaRNG) * (1.0f / sampleCount); // We compute the current pixel color given a ray and the scene data
    }

	accumBuffer[pixelIndex] += pixelColor; // Add the computed color of the current pixel to the accumulation buffer

    GLColor finalColor; // Convert the computed color to a format suitable for OpenGL (24-bits float, i.e. 4 bytes)
    finalColor.colorComponents = make_uchar4((unsigned char)(hdrToSGRB(accumBuffer[pixelIndex].x / frameCounter)),
											(unsigned char)(hdrToSGRB(accumBuffer[pixelIndex].y / frameCounter)),
											(unsigned char)(hdrToSGRB(accumBuffer[pixelIndex].z / frameCounter)),
											1);

    dataHost[pixelIndex] = make_float3(pixelX, pixelY, finalColor.colorValue);
}


extern "C"
void lumenRender(float3 *outputBuffer, float3 *accumBuffer, int renderWidth, int renderHeight,
				int renderSample, int renderBounces, int sphereCount, SphereObject* spheresList,
				int frameCounter)
{
    dim3 cudaThreadsBlock(16, 16, 1);
    dim3 cudaBlocksGrid(renderWidth / cudaThreadsBlock.x, renderHeight / cudaThreadsBlock.y, 1);

    renderDispatcher <<< cudaBlocksGrid, cudaThreadsBlock >>>(outputBuffer, accumBuffer, renderWidth, renderHeight, renderSample, renderBounces, sphereCount, spheresList, frameCounter);
}
