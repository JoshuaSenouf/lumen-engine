#ifndef RENDERKERNEL_H
#define RENDERKERNEL_H

#include "object.h"
#include "camera.h"


extern "C"
void lumenRender(glm::vec3 *outputBuffer, glm::vec3 *accumBuffer, int renderWidth, int renderHeight,
                int renderSample, int renderBounces, int sphereCount, SphereObject* spheresList,
                int frameNumber, CameraInfo* cameraInfo);


#endif // RENDERKERNEL_H
