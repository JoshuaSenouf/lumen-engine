#ifndef CUDARENDERER_H
#define CUDARENDERER_H

#include <tuple>

#include "cuda_runtime.h"
#include "src/scene/sceneParser.h"
#include "src/cuda/cutil_math.h"
#include "src/cuda/renderKernel.h"


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



class cudaRenderer
{
    public:
        cudaRenderer();
        void setScene();
        void render(int width, int height, int samples, int bounces);

    private:
        SphereObject* spheres;
        int sphereCount;
};

#endif // CUDARENDERER_H
