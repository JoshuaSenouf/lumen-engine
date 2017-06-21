#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <vector_types.h>

#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"
#include "thrust\iterator\zip_iterator.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "math_helper.h"
#include "renderKernel.h"
#include "material.h"
#include "object.h"


#define EPSILON 0.0001f
#define LIGHT_INTENSITY 2.0f
#define METAL_EXPO 30.0f
#define GLOSSY_LEVEL 0.1f


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


__device__ RayObject getCameraRay(unsigned int posX, unsigned int posY, CameraInfo cameraInfo, curandState *cudaRNG) // Inspired a lot by Peter Kutz's path tracer
{
    glm::vec3 horizontalAxis = normalize(cross(cameraInfo.cameraFront, cameraInfo.cameraUp));
    glm::vec3 verticalAxis = normalize(cross(horizontalAxis, cameraInfo.cameraFront));

    glm::vec3 middle = cameraInfo.cameraPosition + cameraInfo.cameraFront;
    glm::vec3 horizontal = horizontalAxis * tanf(cameraInfo.cameraFOV.x * 0.5f * (M_PI / 180));
    glm::vec3 vertical = verticalAxis * tanf(cameraInfo.cameraFOV.y * -0.5f * (M_PI / 180));

    float rayJitterX = ((curand_uniform(cudaRNG) - 0.5f) + posX) / (cameraInfo.cameraResolution.x - 1.0f);
    float rayJitterY = ((curand_uniform(cudaRNG) - 0.5f) + posY) / (cameraInfo.cameraResolution.y - 1.0f);

    glm::vec3 cameraPointOnPlane = cameraInfo.cameraPosition
                                + ((middle
                                + (horizontal * ((2.0f * rayJitterX) - 1.0f))
                                + (vertical * ((2.0f * rayJitterY) - 1.0f))
                                - cameraInfo.cameraPosition)
                                * cameraInfo.cameraFocalDistance);

    glm::vec3 cameraAperturePoint = cameraInfo.cameraPosition;

    if (cameraInfo.cameraApertureRadius > 0.0f)
    {
        float randomizedAngle = 2.0f * M_PI * curand_uniform(cudaRNG);
        float randomizedRadius = cameraInfo.cameraApertureRadius * sqrt(curand_uniform(cudaRNG));
        float apertureX = cosf(randomizedAngle) * randomizedRadius;
        float apertureY = sinf(randomizedAngle) * randomizedRadius;

        cameraAperturePoint = cameraInfo.cameraPosition + (horizontalAxis * apertureX) + (verticalAxis * apertureY);
    }

    glm::vec3 rayOrigin = cameraAperturePoint;
    glm::vec3 rayDirection = normalize(cameraPointOnPlane - cameraAperturePoint);

    return RayObject(rayOrigin, rayDirection);
}


__device__ float checkSphereIntersect(SphereObject sphere, RayObject ray)
{
    glm::vec3 op = sphere.position - ray.origin;
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


__device__ glm::vec3 computeCosineWeightedImportanceSampling(glm::vec3 localW, glm::vec3 localU, glm::vec3 localV, float rand1, float rand2, float cosT)
{
    return normalize(localU * cos(rand1) * rand2 + localV * sin(rand1) * rand2 + localW * cosT);

}


__device__ glm::vec3 computePerfectlyReflectedRay(glm::vec3 rayDirection, glm::vec3 intersectionNormal)
{
    return rayDirection - 2.0f * intersectionNormal * dot(intersectionNormal, rayDirection);
}


__device__ thrust::tuple<glm::vec3, glm::vec3> computeDiffuseMaterial(glm::vec3 hitOrientedNormal, glm::vec3 hitCoord, float rand1, float rand2, float cosT)
{
    glm::vec3 localOrthoW = hitOrientedNormal;
    glm::vec3 localOrthoU = normalize(cross((fabs(localOrthoW.x) > 0.1f ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f)), localOrthoW));
    glm::vec3 localOrthoV = cross(localOrthoW, localOrthoU);

    // Cosine Weighted Importance Sampling
    glm::vec3 nextRayDir = computeCosineWeightedImportanceSampling(localOrthoW, localOrthoU, localOrthoV, rand1, rand2, cosT);
    hitCoord += hitOrientedNormal * EPSILON;

    return thrust::make_tuple(nextRayDir, hitCoord);
}


__device__ thrust::tuple<glm::vec3, glm::vec3> computePerfectSpecularMaterial(glm::vec3 rayDir, glm::vec3 hitCoord, glm::vec3 hitNormal, glm::vec3 hitOrientedNormal)
{
    glm::vec3 nextRayDir = computePerfectlyReflectedRay(rayDir, hitNormal);
    hitCoord += hitOrientedNormal * EPSILON;

    return thrust::make_tuple(nextRayDir, hitCoord);
}


__device__ thrust::tuple<glm::vec3, glm::vec3> computePhongMetalMaterial(glm::vec3 rayDir, glm::vec3 hitCoord, glm::vec3 hitNormal, float rand1, float sinT, float cosT)
{
    glm::vec3 localOrthoW = normalize(rayDir - hitNormal * 2.0f * dot(hitNormal, rayDir));
    glm::vec3 localOrthoU = normalize(cross((fabs(localOrthoW.x) > 0.1f ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f)), localOrthoW));
    glm::vec3 localOrthoV = cross(localOrthoW, localOrthoU);

    glm::vec3 nextRayDir = computeCosineWeightedImportanceSampling(localOrthoW, localOrthoU, localOrthoV, rand1, sinT, cosT);
    hitCoord += localOrthoW * EPSILON;

    return thrust::make_tuple(nextRayDir, hitCoord);
}


__device__ glm::vec3 computeRadiance(RayObject &cameraRay, int sphereCount, SphereObject* spheresList, int lightBounces, curandState *cudaRNG)
{
    glm::vec3 colorAccumulation = glm::vec3(0.0f);
    glm::vec3 colorMask = glm::vec3(1.0f);

    for (int bounces = 0; bounces < lightBounces; bounces++)
    {
        float closestSphereDist;
        int closestSphereID = 0;

        if (!checkSceneIntersect(cameraRay, sphereCount, spheresList, closestSphereDist, closestSphereID))
            return colorAccumulation += colorMask * glm::vec3(0.7f, 0.8f, 0.8f); // If we hit no object, we return the sky color

        const SphereObject &hitSphere = spheresList[closestSphereID];
        glm::vec3 hitCoord = cameraRay.origin + cameraRay.direction * closestSphereDist;
        glm::vec3 hitNormal = normalize(hitCoord - hitSphere.position);
        glm::vec3 hitOrientedNormal = dot(hitNormal, cameraRay.direction) < 0.0f ? hitNormal : hitNormal * -1.0f;

        colorAccumulation += colorMask * hitSphere.emissiveColor;

        glm::vec3 nextRayDir;

        // DIFFUSE
        if (hitSphere.material == 1)
        {
            float curand1 = 2.0f * M_PI * curand_uniform(cudaRNG);
            float curand2 = curand_uniform(cudaRNG);
            float curand2Square = sqrtf(curand2);
            float cosT = sqrtf(1.0f - curand2);

            thrust::tie(nextRayDir, hitCoord) = computeDiffuseMaterial(hitOrientedNormal, hitCoord, curand1, curand2Square, cosT);

            colorMask *= dot(nextRayDir, hitOrientedNormal);
            colorMask *= hitSphere.color;
        }

        // (PERFECT) SPECULAR
        else if (hitSphere.material == 2)
        {
            thrust::tie(nextRayDir, hitCoord) = computePerfectSpecularMaterial(cameraRay.direction, hitCoord, hitNormal, hitOrientedNormal);

            colorMask *= hitSphere.color;
        }

        // REFRACT
        else if (hitSphere.material == 3)
        {

        }

        // METAL (PHONG)
        else if (hitSphere.material == 4)
        {
            float curand1 = 2.0f * M_PI * curand_uniform(cudaRNG);
            float curand2 = curand_uniform(cudaRNG);
            float cosTMetal = powf(1.0f - curand2, 1.0f / (METAL_EXPO + 1.0f));
            float sinTMetal = sqrtf(1.0f - cosTMetal * cosTMetal);

            thrust::tie(nextRayDir, hitCoord) = computePhongMetalMaterial(cameraRay.direction, hitCoord, hitNormal, curand1, sinTMetal, cosTMetal);

            colorMask *= hitSphere.color;
        }

        // GLOSSY/COAT (from Peter Kurtz path tracer, not physically accurate but nice to have anyway)
        else if (hitSphere.material == 5)
        {
            float curand1 = curand_uniform(cudaRNG);
            bool materialSpecular = (curand1 < GLOSSY_LEVEL);

            // We simply choose between computing a perfect specular or a diffuse material depending of the random value when compared to a certain threshold of glossiness
            if (materialSpecular)
            {
                thrust::tie(nextRayDir, hitCoord) = computePerfectSpecularMaterial(cameraRay.direction, hitCoord, hitNormal, hitOrientedNormal);

                colorMask *= hitSphere.color;
            }

            else
            {
                float curand1 = 2.0f * M_PI * curand_uniform(cudaRNG);
                float curand2 = curand_uniform(cudaRNG);
                float curand2Square = sqrtf(curand2);
                float cosT = sqrtf(1.0f - curand2);

                thrust::tie(nextRayDir, hitCoord) = computeDiffuseMaterial(hitOrientedNormal, hitCoord, curand1, curand2Square, cosT);

                colorMask *= dot(nextRayDir, hitOrientedNormal);
                colorMask *= hitSphere.color;
            }
        }

        //colorMask *= LIGHT_INTENSITY;

        cameraRay.direction = nextRayDir;
        cameraRay.origin = hitCoord;
    }

    return colorAccumulation;
}


__global__ void renderDispatcher(glm::vec3 *dataHost, glm::vec3* accumBuffer, int renderWidth, int renderHeight,
                                int sampleCount, int lightBounces, int sphereCount, SphereObject *spheresList,
                                int frameCounter, CameraInfo* cameraInfo)
{
    unsigned int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int pixelY = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int pixelIndex = (renderHeight - pixelY - 1) * renderWidth + pixelX;
    unsigned int posX = pixelX;
    unsigned int posY = renderHeight - pixelY - 1;

    int threadIndex = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    curandState cudaRNG;
    curand_init(WangHash(frameCounter) + threadIndex, 0, 0, &cudaRNG); // We create a new seed using curand and our hashed framecounter

    glm::vec3 pixelColor;
    pixelColor = glm::vec3(0.0f);

    for (int sample = 0; sample < sampleCount; sample++)
    {
        RayObject cameraRay = getCameraRay(posX, posY, *cameraInfo, &cudaRNG);

        pixelColor += computeRadiance(cameraRay, sphereCount, spheresList, lightBounces, &cudaRNG) * (1.0f / sampleCount); // We compute the current pixel color given a ray from the camera and the scene data
    }

    accumBuffer[pixelIndex] += pixelColor; // Add the computed color of the current pixel to the accumulation buffer

    GLColor finalColor; // Convert the computed color to a format suitable for OpenGL (24-bits float, i.e. 4 bytes)
    finalColor.colorComponents = make_uchar4((unsigned char)(hdrToSGRB(accumBuffer[pixelIndex].x / frameCounter)),
                                            (unsigned char)(hdrToSGRB(accumBuffer[pixelIndex].y / frameCounter)),
                                            (unsigned char)(hdrToSGRB(accumBuffer[pixelIndex].z / frameCounter)),
                                            1);

    dataHost[pixelIndex] = glm::vec3(pixelX, pixelY, finalColor.colorValue);
}


extern "C"
void lumenRender(glm::vec3 *outputBuffer, glm::vec3 *accumBuffer, int renderWidth, int renderHeight,
                int renderSample, int renderBounces, int sphereCount, SphereObject* spheresList,
                int frameCounter, CameraInfo* cameraInfo)
{
    dim3 cudaThreadsBlock(8, 8);
    dim3 cudaBlocksGrid(renderWidth / cudaThreadsBlock.x, renderHeight / cudaThreadsBlock.y);

    renderDispatcher <<< cudaBlocksGrid, cudaThreadsBlock >>>(outputBuffer, accumBuffer, renderWidth, renderHeight, renderSample, renderBounces, sphereCount, spheresList, frameCounter, cameraInfo);
}
