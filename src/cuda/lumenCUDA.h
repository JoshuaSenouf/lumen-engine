#ifndef LUMENCUDA_H
#define LUMENCUDA_H

#include "cuda_runtime.h"

#include "object.h"
#include "scene.h"
#include "renderKernel.h"
#include "cutil_math.h"


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
