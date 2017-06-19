#ifndef RENDERKERNEL_H
#define RENDERKERNEL_H

#include "object.h"


extern "C"
void lumenRender(glm::vec3 *outputBuffer, glm::vec3 *accumBuffer, int renderWidth, int renderHeight, int renderSample, int renderBounces, int sphereCount, SphereObject* spheresList, int frameNumber);


#endif // RENDERKERNEL_H
