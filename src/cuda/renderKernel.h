#ifndef RENDERKERNEL_H
#define RENDERKERNEL_H

#include "object.h"


extern "C"
void lumenRender(float3 *outputBuffer, float3 *accumBuffer, int renderWidth, int renderHeight, int renderSample, int renderBounces, int sphereCount, SphereObject* spheresList, int frameNumber);

#endif // RENDERKERNEL_H
