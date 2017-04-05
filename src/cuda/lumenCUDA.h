#ifndef LUMENCUDA_H
#define LUMENCUDA_H

#include "cuda_runtime.h"
#include "helper_cuda.h"

#include "src/scene/object.h"
#include "src/scene/scene.h"
#include "src/cuda/renderKernel.h"
#include "api/cuda/cutil_math.h"


class LumenCUDA
{
    public:
        LumenCUDA();

        void setScene();
        void render(int width, int height, int samples, int bounces);

    private:
        SphereObject* spheres;
        int sphereCount;
};

#endif // LUMENCUDA_H
