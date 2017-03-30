#ifndef RENDERKERNEL_H
#define RENDERKERNEL_H


struct SphereObject;


void lumenRender(int width, int height, int samples, int bounces, int sphereCount, SphereObject* spheres);


#endif // RENDERKERNEL_H
