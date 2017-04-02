#ifndef CUDARENDERER_H
#define CUDARENDERER_H

#include "src/scene/object.h"
#include "src/scene/scene.h"
#include "src/cuda/cutil_math.h"
#include "src/cuda/renderKernel.h"
#include "cuda_runtime.h"
#include "helper_cuda.h"


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
